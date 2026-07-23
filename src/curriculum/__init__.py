"""Curriculum learning for GraphQA meta-ICL training.

Public entry point ``build_curriculum`` wires together difficulty scoring and
the competence-based sampler from a Hydra config and a training dataset.
"""

import math
import os

from .difficulty import (
    combined_sample_difficulty,
    difficulty_to_cdf,
    within_group_cdf,
    load_or_compute_task_difficulty,
)
from .sampler import CompetenceCurriculumSampler, CurriculumCallback

__all__ = [
    "build_curriculum",
    "CompetenceCurriculumSampler",
    "CurriculumCallback",
    "load_or_compute_task_difficulty",
]


def _group_key(record, dims):
    return "|".join(str(record.get(d, "unknown")) for d in dims)


def build_curriculum(cfg, train_dataset, wandb_dir="wandb"):
    """Build the curriculum sampler and its Lightning callback from config.

    Returns ``(sampler, callback)``. The caller passes ``sampler`` to the train
    DataLoader (with ``shuffle=False``) and appends ``callback`` to the trainer
    callbacks so competence tracks the global step.
    """
    cur = cfg.curriculum
    records = train_dataset.records

    # 1. Task-level difficulty from OOD transfer results (optional).
    if cur.task_difficulty_source == "ood":
        cache_path = cur.get("task_difficulty_file", None)
        task_difficulty = load_or_compute_task_difficulty(
            cache_path=cache_path,
            wandb_dir=wandb_dir,
            metric=cur.ood_difficulty_metric,
            run_name_prefix=cur.get("ood_run_name_prefix", None),
        )
    else:
        task_difficulty = {}

    # 2. Blend with per-sample graph complexity and rank into a CDF.
    features = tuple(cur.graph_features)
    sample_difficulty = combined_sample_difficulty(
        records,
        task_difficulty,
        alpha=cur.alpha_task_vs_graph,
        features=features,
    )

    # Eligibility scope decides how competence unlocks samples:
    #   - "eligible" (default): global ranking -> easiest samples overall unlock
    #     first, so easy tasks appear before hard ones.
    #   - "global": within-task ranking -> the easiest fraction of *every* task
    #     unlocks together, so all tasks are present from step 0 and each ramps
    #     easy->hard internally.
    diversity_scope = cur.get("diversity_scope", "eligible")
    if diversity_scope == "global":
        task_keys = [r.get("task", "unknown") for r in records]
        cdf = within_group_cdf(sample_difficulty, task_keys)
    else:
        cdf = difficulty_to_cdf(sample_difficulty)

    # 3. Diversity grouping keys (task, or task+algorithm).
    dims = list(cur.diversity_dims)
    group_keys = [_group_key(r, dims) for r in records]
    group_id_to_name = {i: name for i, name in enumerate(sorted(set(group_keys)))}

    # 4. Total optimizer steps until competence reaches 1.0.
    batch_size = cfg.dataset.batch_size
    grad_accum = cfg.trainer.get("gradient_accumulation", 1) or 1
    num_batches = math.ceil(len(records) / batch_size)
    steps_per_epoch = max(1, math.ceil(num_batches / grad_accum))
    total_training_steps = steps_per_epoch * cfg.trainer.max_epochs
    total_steps = max(1, int(cur.full_competence_fraction * total_training_steps))

    sampler = CompetenceCurriculumSampler(
        difficulty_cdf=cdf,
        group_keys=group_keys,
        num_samples=len(records),
        total_steps=total_steps,
        c0=cur.c0,
        p=cur.p,
        diversity_weight=cur.diversity_weight,
        sampling=cur.get("sampling", "with_replacement"),
        seed=cur.get("seed", 42),
    )
    callback = CurriculumCallback(sampler, group_id_to_name=group_id_to_name)
    return sampler, callback
