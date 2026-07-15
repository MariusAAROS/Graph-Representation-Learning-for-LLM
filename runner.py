import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer
from src.datasets.loaders import (
    MetaICLDataset,
    BaselineDataset,
    make_collator,
    read_json,
    filter_records_by_task,
    split_holdout_task,
)
from src.models.meta_icl import MetaICL


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
    BASE_DIR = "data/"
    if cfg.dataset.name == "graphqa":
        if is_loto:
            # Leave-one-task-out (LOTO): choose the held-out task at runtime from
            # the OOD pool (all tasks present). train = every task except the
            # held-out one; the held-out task is split 50/50 into val + test.
            pool_records = []
            for split in ["train", "val", "test"]:
                pool_path = os.path.join(BASE_DIR, "graphqa",
                                         f"{cfg.dataset.dataset_config}",
                                         "ood_pool", f"{split}.json")
                if not os.path.exists(pool_path):
                    raise FileNotFoundError(
                        f"OOD pool file not found: {pool_path}. "
                        f"Generate it by running dataset_generator.py with OOD_POOL_MODE=True."
                    )
                pool_records.extend(read_json(pool_path))

            available_tasks = sorted({r["task"] for r in pool_records})
            if ood_task not in available_tasks:
                raise ValueError(
                    f"ood_task '{ood_task}' not found in pool. Available tasks: {available_tasks}"
                )

            train_records = filter_records_by_task(pool_records, exclude_task=ood_task)
            val_records, test_records = split_holdout_task(pool_records, ood_task)
            print(f"[LOTO] held-out task : {ood_task}")
            print(f"[LOTO] train tasks   : {sorted({r['task'] for r in train_records})}")
            print(f"[LOTO] sample counts : train={len(train_records)} "
                  f"val={len(val_records)} test={len(test_records)}")

            if cfg.dataset.dataset_config == "meta-icl":
                train_dataset = MetaICLDataset._from_records(train_records, k=cfg.dataset.n_examples)
                val_dataset   = MetaICLDataset._from_records(val_records, k=cfg.dataset.n_examples)
            elif cfg.dataset.dataset_config == "baseline":
                train_dataset = BaselineDataset._from_records(train_records)
                val_dataset   = BaselineDataset._from_records(val_records)
            else:
                raise ValueError(f"Unknown dataset config: {cfg.dataset.dataset_config}")
        else:
            paths = {}
            for split in ["train", "val", "test"]:
                current_path = os.path.join(BASE_DIR, "graphqa",
                                            f"{cfg.dataset.dataset_config}",
                                            f"{cfg.dataset.test_type}",
                                            f"{split}.json")
                if os.path.exists(current_path):
                    paths[split] = current_path
                else:
                    raise FileNotFoundError(f"File not found: {current_path}")

            if cfg.dataset.dataset_config == "meta-icl":
                train_dataset = MetaICLDataset(paths["train"], k=cfg.dataset.n_examples)
                val_dataset   = MetaICLDataset(paths["val"], k=cfg.dataset.n_examples)
            elif cfg.dataset.dataset_config == "baseline":
                train_dataset = BaselineDataset(paths["train"])
                val_dataset   = BaselineDataset(paths["val"])
            else:
                raise ValueError(f"Unknown dataset config: {cfg.dataset.dataset_config}")

        model = MetaICL(cfg)
        collator = make_collator(
            tokenizer=AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True),
            max_length=cfg.model.max_seq_len,
            padding_side="right"
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

    # batch_size = n_tasks_per_batch: each item is one episode (one task)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
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

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, early_stop_cb],
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.gradient_accumulation,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        wandb.finish()   # close this run so the next multirun job starts a fresh one

if __name__ == "__main__":
    train()
