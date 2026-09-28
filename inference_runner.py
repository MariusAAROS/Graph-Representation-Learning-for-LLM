import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.datasets.loaders import make_collator, build_graphqa_datasets
from src.models.meta_icl import MetaICL
from src.models.hrm_text import (
    HRMGraphICL,
    HRMTextICL,
    make_hrm_collator,
    make_structure_fn,
)
from src.utils import model_slug


@hydra.main(config_path="configs", config_name="baseline.yaml", version_base="1.2")
def infer(cfg: DictConfig):
    """Inference-only runner: evaluate a model on the TEST split without training.

    Primary use is ablations with a raw (untrained) model: when
    `inference.checkpoint_path` is null, the pretrained backbone is loaded but
    never fine-tuned, so evaluating it isolates the contribution of the SFT.
    Pass a checkpoint to score a trained model on the exact same test split with
    the identical teacher-forcing metric used during training validation.
    """
    ood_task = cfg.dataset.get("ood_task", None)
    is_loto = cfg.dataset.test_type == "ood" and ood_task is not None

    checkpoint_path = cfg.inference.get("checkpoint_path", None)
    run_tag = cfg.inference.get("run_tag", "raw")

    base_name = cfg.logger.name
    if cfg.logger.get("include_model_tag", False):
        base_name = f"{base_name}-{model_slug(cfg.model.name)}"

    if is_loto:
        logger_name = f"{base_name}-{run_tag}-ood-{ood_task}"
    else:
        suffix = "id" if cfg.dataset.test_type == "standard" else "ood"
        logger_name = f"{base_name}-{run_tag}-{suffix}"

    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=logger_name,
        group=f"{base_name}-{run_tag}",  # group ablation runs together
        reinit=True,
    )

    # Evaluate on the held-out TEST split only.
    datasets = build_graphqa_datasets(cfg, splits=("test",))
    test_dataset = datasets["test"]

    if is_loto:
        print(f"[LOTO/infer] held-out task : {ood_task}")
        print(f"[LOTO/infer] test samples  : {len(test_dataset)}")

    # Load a trained checkpoint, or fall back to the raw (untrained) model.
    # Tag logger.name so the model's predictions/ folder is distinguishable from
    # trained runs on disk (MetaICL derives its predictions dir from this name).
    cfg.logger.name = f"{base_name}-{run_tag}"
    arch = cfg.model.get("arch", "causal_lm")
    ModelCls = {"hrm_text": HRMTextICL, "hrm_graph": HRMGraphICL}.get(arch, MetaICL)
    if checkpoint_path:
        print(f"[infer] loading checkpoint: {checkpoint_path}")
        model = ModelCls.load_from_checkpoint(checkpoint_path, config=cfg)
    else:
        print("[infer] no checkpoint -> using raw (untrained) model as ablation")
        model = ModelCls(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    if arch in ("hrm_text", "hrm_graph"):
        collator = make_hrm_collator(
            tokenizer=tokenizer,
            max_length=cfg.model.max_seq_len,
            padding_side="right",
            condition=cfg.model.get("condition", ""),
            structure_fn=make_structure_fn(cfg, train=False),
        )
    else:
        collator = make_collator(
            tokenizer=tokenizer,
            max_length=cfg.model.max_seq_len,
            padding_side="right",
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )

    trainer = pl.Trainer(
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    try:
        # Reuse the model's validation loop (teacher-forcing exact-match +
        # per-task metrics + predictions CSV) to stay comparable with training.
        trainer.validate(model, dataloaders=test_loader)
    finally:
        wandb.finish()


if __name__ == "__main__":
    infer()
