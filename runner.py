import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.datasets.loaders import (
    make_collator,
    build_graphqa_datasets,
)
from src.models.meta_icl import MetaICL
from src.curriculum import build_curriculum


@hydra.main(config_path="configs", config_name="baseline.yaml", version_base="1.2")
def train(cfg: DictConfig):
    ood_task = cfg.dataset.get("ood_task", None)
    is_loto = cfg.dataset.test_type == "ood" and ood_task is not None

    if is_loto:
        logger_name = f"{cfg.logger.name}-ood-{ood_task}"
    else:
        logger_name = f"{cfg.logger.name}-{'id' if cfg.dataset.test_type == 'standard' else 'ood'}"
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=logger_name,
        group=cfg.logger.name,   # group LOTO runs together for side-by-side comparison
        reinit=True,             # force a new run per Hydra multirun job (same process)
    )

    datasets = build_graphqa_datasets(cfg, splits=("train", "val"))
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    if is_loto:
        print(f"[LOTO] held-out task : {ood_task}")
        print(f"[LOTO] train tasks   : {sorted({r['task'] for r in train_dataset.records})}")
        print(f"[LOTO] sample counts : train={len(train_dataset)} "
              f"val={len(val_dataset)}")

    model = MetaICL(cfg)
    collator = make_collator(
        tokenizer=AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True),
        max_length=cfg.model.max_seq_len,
        padding_side="right"
    )

    # Curriculum learning: replace random shuffling with a competence-based
    # sampler that unlocks harder samples as training progresses, while keeping
    # task diversity in each batch. Disabled by default (plain shuffling).
    curriculum_enabled = cfg.get("curriculum", {}).get("enabled", False)
    curriculum_sampler = None
    curriculum_callback = None
    if curriculum_enabled:
        curriculum_sampler, curriculum_callback = build_curriculum(cfg, train_dataset)
        print(f"[curriculum] enabled | total_steps_to_full_competence="
              f"{curriculum_sampler.total_steps} | c0={curriculum_sampler.c0} "
              f"p={curriculum_sampler.p} diversity={curriculum_sampler.diversity_weight}")

    # batch_size = n_tasks_per_batch: each item is one episode (one task)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=curriculum_sampler is None,
        sampler=curriculum_sampler,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.trainer.checkpoint_monitor,
        mode=cfg.trainer.checkpoint_mode,
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val/exact_match:.3f}",
    )
    early_stop_cb = EarlyStopping(
        monitor=cfg.trainer.early_stopping_monitor,
        patience=cfg.trainer.early_stopping_patience,
        mode=cfg.trainer.early_stopping_mode,
    )

    callbacks = [checkpoint_cb, early_stop_cb]
    if curriculum_callback is not None:
        callbacks.append(curriculum_callback)

    # The "eligible_only" curriculum sampler yields a growing number of samples
    # per epoch, so Lightning must re-query the dataloader length each epoch
    # instead of caching the epoch-0 value.
    reload_every = 0
    if curriculum_enabled and cfg.curriculum.get("sampling", "with_replacement") == "eligible_only":
        reload_every = 1

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.gradient_accumulation,
        reload_dataloaders_every_n_epochs=reload_every,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        wandb.finish()   # close this run so the next multirun job starts a fresh one

if __name__ == "__main__":
    train()
