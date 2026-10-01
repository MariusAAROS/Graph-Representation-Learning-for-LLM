import os
# Reduce CUDA caching-allocator fragmentation from variable-length batches.
# Must be set before any CUDA context is created, hence before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import time

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.datasets.loaders import build_graphqa_datasets
from src.models.hrm_text import HRMTextICL, make_hrm_collator


@hydra.main(config_path="configs", config_name="hrm_text.yaml", version_base="1.2")
def run(cfg: DictConfig):
    """Evaluate one HRM-Text checkpoint at several recursion depths (H_cycles, L_cycles).

    HrmTextModel reads H_cycles / L_cycles from its config on every forward, and the
    layers per stack do not depend on them, so the same weights run at any depth. Each
    depth is scored with the training-time validation loop (teacher-forced exact match,
    per-task metrics, majority baseline), as inference_runner.py does, and logged as its
    own W&B run. The logged keys stay `val/*`; the run config records the split.

        python depth_runner.py dataset.name=graphqa dataset.dataset_config=baseline \
            dataset.test_type=standard +depth.ckpt=<last.ckpt> \
            '+depth.configs=[[1,1],[2,3]]' +depth.tag=graphqa logger.project=<project>

    depth.ckpt: Lightning checkpoint of HRMTextICL (null = raw pretrained model).
    depth.configs: list of [H, L]; duplicates are dropped.
    depth.split: test (default) or val.
    depth.tag: run-name prefix; runs are named <exp>-<tag>-H{H}L{L}.
    depth.exp: experiment id, also the W&B group (default "b").
    depth.trained: [H, L] the checkpoint was trained at (default: its loaded config).
    depth.extra: flat dict merged into each run's W&B config.
    depth.limit_val_batches: optional, for smoke tests.
    """
    depth = cfg.depth
    split = depth.get("split", "test")
    exp = depth.get("exp", "b")
    tag = depth.get("tag", cfg.dataset.name)
    ckpt = depth.get("ckpt", None)
    configs = list(dict.fromkeys(tuple(int(v) for v in c) for c in depth.configs))

    if cfg.dataset.test_type == "ood":
        raise ValueError("depth_runner evaluates standard splits only; pass dataset.test_type=standard")
    dataset = build_graphqa_datasets(cfg, splits=(split,))[split]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    collator = make_hrm_collator(
        tokenizer=tokenizer,
        max_length=cfg.model.max_seq_len,
        padding_side="right",
        condition=cfg.model.get("condition", ""),
    )
    # Batches are built in the main process: this script opens one W&B run per depth, and
    # DataLoader workers forked while a run is open broke its service connection and hung the
    # process. Tokenizing a batch is negligible next to the forward pass.
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    model = HRMTextICL(cfg)
    if ckpt:
        # Weights only: Lightning's load_from_checkpoint goes through torch.load(weights_only=True),
        # which rejects the OmegaConf hyperparameters stored in these checkpoints.
        print(f"[depth] loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
        del state
    else:
        print("[depth] no checkpoint -> raw pretrained model")
    model.eval()

    hf_config = model.model.config
    trained = tuple(depth.get("trained", None) or (hf_config.H_cycles, hf_config.L_cycles))
    extra = OmegaConf.to_container(depth.extra, resolve=True) if depth.get("extra", None) else {}
    predictions_root = os.path.join(cfg.trainer.predictions_dir, f"depth-{exp}-{tag}")

    for H, L in configs:
        hf_config.H_cycles, hf_config.L_cycles = H, L
        name = f"{exp}-{tag}-H{H}L{L}"
        model._predictions_dir = os.path.join(predictions_root, f"H{H}L{L}")
        os.makedirs(model._predictions_dir, exist_ok=True)

        logger = WandbLogger(
            project=cfg.logger.project,
            name=name,
            group=exp,
            reinit=True,
            config={
                "exp": exp, "dataset": cfg.dataset.name, "split": split,
                "H": H, "L": L, "trained_H": trained[0], "trained_L": trained[1],
                "is_trained_depth": (H, L) == trained, "source_ckpt": ckpt or "pretrained",
                **extra,
            },
        )
        trainer_kwargs = dict(
            precision=cfg.trainer.precision,
            accelerator=cfg.trainer.accelerator,
            logger=logger,
            log_every_n_steps=10,
            enable_checkpointing=False,
        )
        if depth.get("limit_val_batches", None) is not None:
            trainer_kwargs["limit_val_batches"] = depth.limit_val_batches
        trainer = pl.Trainer(**trainer_kwargs)

        try:
            start = time.time()
            trainer.validate(model, dataloaders=loader)
            elapsed = time.time() - start
            # Per-sample predictions written by the validation loop, kept on W&B as well.
            with open(os.path.join(model._predictions_dir, "predictions_epoch_0.csv"), newline="") as f:
                rows = list(csv.reader(f))
            logger.experiment.log({"predictions": wandb.Table(columns=rows[0], data=rows[1:]),
                                   "eval/seconds": elapsed})
            print(f"[depth] {name}: {elapsed:.0f}s")
        finally:
            wandb.finish()


if __name__ == "__main__":
    run()
