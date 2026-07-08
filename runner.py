import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer
from src.datasets.loaders import MetaICLDataset, BaselineDataset, make_collator
from src.models.meta_icl import MetaICL


@hydra.main(config_path="configs", config_name="meta_icl.yaml", version_base="1.2")
def train(cfg: DictConfig):
    logger_name = f"{cfg.logger.name}-{'id' if cfg.dataset.test_type == 'standard' else 'ood'}"
    wandb_logger = WandbLogger(project=cfg.logger.project, name=logger_name)
    BASE_DIR = "data/"
    if cfg.dataset.name == "graphqa":
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
        # gradient_clip_val=cfg.trainer.gradient_clip_val,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()
