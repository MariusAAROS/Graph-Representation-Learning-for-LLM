"""Competence-based curriculum sampler with a diversity factor.

Implements the competence pacing of Platanios et al. (2019), "Competence-based
Curriculum Learning for Neural Machine Translation": a competence value
``c(t)`` grows from ``c0`` to 1 over training, and at step ``t`` only samples
whose CDF-difficulty is ``<= c(t)`` are eligible. Eligible samples are drawn
with replacement so the number of optimizer steps per epoch stays constant
regardless of how much of the dataset is currently unlocked (or, in
``eligible_only`` mode, in a single pass so early epochs are short and grow).

A diversity factor interpolates between difficulty-faithful uniform sampling and
group-balanced sampling (by task, and optionally graph-generator family), so
each batch keeps a mix of tasks instead of collapsing onto whichever tasks
happen to be easiest early in training.
"""

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Sampler


class CompetenceCurriculumSampler(Sampler):
    """Sample indices according to a growing competence and a diversity factor.

    Args:
        difficulty_cdf: Per-sample difficulty mapped to a CDF in (0, 1].
        group_keys: Per-sample grouping key used for diversity balancing
            (e.g. the task name, or "task|algorithm").
        num_samples: Number of indices to yield per epoch (typically
            ``len(dataset)``).
        total_steps: Optimizer step at which competence reaches 1.0.
        c0: Initial competence in (0, 1]; the fraction of (easiest) data
            available at step 0.
        p: Pacing exponent. ``p=2`` (sqrt schedule) spends longer on easy
            samples; ``p=1`` is linear.
        diversity_weight: 0 = uniform over eligible samples (difficulty
            faithful); 1 = fully group-balanced. Values in between interpolate.
        sampling: "with_replacement" draws ``num_samples`` eligible indices with
            replacement (constant-length epochs, diversity-weighted).
            "eligible_only" yields one shuffled pass over the unlocked pool
            (no oversampling; epoch length grows with competence).
        seed: Base RNG seed; combined with the epoch for per-epoch shuffling.
    """

    def __init__(
        self,
        difficulty_cdf,
        group_keys,
        num_samples,
        total_steps,
        c0=0.1,
        p=2.0,
        diversity_weight=0.5,
        sampling="with_replacement",
        seed=42,
    ):
        self.difficulty_cdf = np.asarray(difficulty_cdf, dtype=np.float64)
        self.num_samples = int(num_samples)
        self.total_steps = max(1, int(total_steps))
        self.c0 = float(c0)
        self.p = float(p)
        self.diversity_weight = float(diversity_weight)
        self.sampling = sampling
        self.seed = int(seed)

        # Encode group keys as integer ids for fast per-group aggregation.
        unique = {key: i for i, key in enumerate(sorted(set(group_keys)))}
        self.group_ids = np.array([unique[k] for k in group_keys], dtype=np.int64)
        self.num_groups = len(unique)

        # Mutable state, updated by CurriculumCallback each training batch.
        self.global_step = 0
        self.current_epoch = 0
        # Diagnostics filled during __iter__ for logging.
        self.last_competence = self.competence(0)
        self.last_group_counts: dict = {}

    def competence(self, step):
        """Platanios competence: fraction of (easiest) data unlocked at ``step``."""
        c0_p = self.c0 ** self.p
        frac = step * (1.0 - c0_p) / self.total_steps + c0_p
        return float(min(1.0, frac ** (1.0 / self.p)))

    def _sampling_weights(self, eligible):
        """Blend uniform and group-balanced weights over the eligible indices."""
        n_elig = eligible.size
        uniform = np.ones(n_elig, dtype=np.float64)
        if self.diversity_weight <= 0.0 or self.num_groups <= 1:
            return uniform

        elig_groups = self.group_ids[eligible]
        counts = np.bincount(elig_groups, minlength=self.num_groups)
        # Balanced weight: within an eligible group of size m, each sample gets
        # N_eligible / (num_present_groups * m) so every present group carries
        # equal total probability mass.
        present = counts > 0
        num_present = int(present.sum())
        balanced_per_group = np.zeros(self.num_groups, dtype=np.float64)
        balanced_per_group[present] = n_elig / (num_present * counts[present])
        balanced = balanced_per_group[elig_groups]

        lam = self.diversity_weight
        return (1.0 - lam) * uniform + lam * balanced

    def __iter__(self):
        c = self.competence(self.global_step)
        self.last_competence = c

        eligible = np.where(self.difficulty_cdf <= c)[0]
        if eligible.size == 0:
            # Always keep at least the single easiest sample available.
            eligible = np.array([int(np.argmin(self.difficulty_cdf))])

        rng = np.random.default_rng(self.seed + self.current_epoch)
        if self.sampling == "eligible_only":
            # One shuffled pass over the currently-unlocked pool: no oversampling
            # of easy samples, and the epoch length grows as competence rises.
            # diversity_weight is not applied in this mode (each sample once).
            chosen = eligible.copy()
            rng.shuffle(chosen)
        else:  # "with_replacement": constant epoch length, diversity-weighted
            weights = self._sampling_weights(eligible)
            probs = weights / weights.sum()
            chosen = rng.choice(
                eligible, size=self.num_samples, replace=True, p=probs
            )

        elig_groups = self.group_ids[chosen]
        self.last_group_counts = {
            int(g): int(c_) for g, c_ in zip(*np.unique(elig_groups, return_counts=True))
        }
        return iter(chosen.tolist())

    def __len__(self):
        if self.sampling == "eligible_only":
            c = self.competence(self.global_step)
            eligible = np.where(self.difficulty_cdf <= c)[0]
            return max(1, int(eligible.size))
        return self.num_samples


class CurriculumCallback(pl.Callback):
    """Feed the trainer's global step to the sampler and log curriculum stats.

    The sampler evaluates competence once per epoch (when its ``__iter__`` runs),
    so pushing ``trainer.global_step`` before each batch keeps the next epoch's
    competence current. Competence and per-task sampled counts are logged so the
    curriculum progression is visible in wandb.
    """

    def __init__(self, sampler, group_id_to_name=None):
        super().__init__()
        self.sampler = sampler
        self.group_id_to_name = group_id_to_name or {}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.sampler.global_step = trainer.global_step
        self.sampler.current_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return
        metrics = {"curriculum/competence": self.sampler.last_competence}
        for gid, count in self.sampler.last_group_counts.items():
            name = self.group_id_to_name.get(gid, str(gid))
            metrics[f"curriculum/count_{name}"] = count
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
