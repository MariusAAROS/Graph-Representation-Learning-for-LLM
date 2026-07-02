import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader



@hydra.main(config_path="configs", config_name="clever.yaml", version_base="1.2")
def train(cfg: DictConfig):
    wandb_logger = WandbLogger(project=cfg.logger.project, name=cfg.logger.name)

    if cfg.config_name == "<COMPLETE>":
        if cfg.dataset.name == "<COMPLETE>":
            train_dataset = ...
            val_dataset   = ...
        else:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")
        model = ...
        collator = ...
    elif cfg.config_name == "<BASELINE>":
        if cfg.dataset.name == "<COMPLETE>":
            train_dataset = ...
            val_dataset   = ...
        elif cfg.dataset.name == "<COMPLETE>":
            train_dataset = ...
            val_dataset   = ...
        else:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")
        model = ...
        collator = ...
    else:
        raise ValueError(f"Unknown config name: {cfg.config_name}")

    # batch_size = n_tasks_per_batch: each item is one episode (one task)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.n_tasks_per_batch,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.n_tasks_per_batch,
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
