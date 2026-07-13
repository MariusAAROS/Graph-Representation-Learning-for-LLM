from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import os, csv
from datetime import datetime
import pytorch_lightning as pl
import torch

class MetaICL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Warning: No pad_token found for {self.hparams.model.name}, setting pad_token to eos_token ({self.tokenizer.eos_token})")
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(self.hparams.model.name)
        self.model.train()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.hparams.model, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        lora_cfg = getattr(self.hparams, "lora", None)
        if lora_cfg is not None and getattr(lora_cfg, "enabled", False):
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                bias=lora_cfg.bias,
                target_modules=list(lora_cfg.target_modules),
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # Per-run predictions directory, named by timestamp + logger run name
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = getattr(self.hparams.logger, "name", "run")
        self._predictions_dir = os.path.join(
            self.hparams.trainer.predictions_dir, f"{run_ts}_{run_name}"
        )
        os.makedirs(self._predictions_dir, exist_ok=True)
        self._val_buffer: list = []

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        # Causal LM: logits at position t predict token t+1, so shift to align
        # predictions with their target labels before comparing.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        preds = torch.argmax(shift_logits, dim=-1)
        self.log("val/loss", loss, prog_bar=True)

        tasks = batch.get("task", [None] * len(preds))
        for i in range(len(preds)):
            self._val_buffer.append({
                "pred": preds[i].cpu(),
                "label": shift_labels[i].cpu(),
                "loss": loss.item(),
                "task": tasks[i],
            })

    def on_validation_epoch_end(self):
        if not self._val_buffer:
            return
        # Calculate accuracy
        for item in self._val_buffer:
            filtering_mask = item["label"] != -100
            item["pred"] = item["pred"][filtering_mask]
            item["label"] = item["label"][filtering_mask]

        # Global exact-match accuracy
        correct = sum(
            1 for item in self._val_buffer
            if torch.equal(item["pred"], item["label"])
        )
        total = len(self._val_buffer)
        accuracy = correct / total
        self.log("val/exact_match", accuracy, prog_bar=True)

        # Per-task exact-match accuracy (some tasks are harder than others)
        task_correct: dict = {}
        task_total: dict = {}
        for item in self._val_buffer:
            task = item["task"] if item["task"] is not None else "unknown"
            task_total[task] = task_total.get(task, 0) + 1
            if torch.equal(item["pred"], item["label"]):
                task_correct[task] = task_correct.get(task, 0) + 1
        for task in sorted(task_total):
            task_acc = task_correct.get(task, 0) / task_total[task]
            self.log(f"val/exact_match_{task}", task_acc)
            print(
                f"[val] epoch {self.current_epoch} | task={task} | "
                f"exact_match={task_acc:.4f} "
                f"({task_correct.get(task, 0)}/{task_total[task]})"
            )
        print(
            f"[val] epoch {self.current_epoch} | task=ALL | "
            f"exact_match={accuracy:.4f} ({correct}/{total})"
        )

        # Save predictions to CSV
        predictions_file = os.path.join(self._predictions_dir, f"predictions_epoch_{self.current_epoch}.csv")
        with open(predictions_file, "w", newline="") as csvfile:
            fieldnames = ["task", "pred", "label", "loss"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in self._val_buffer:
                decoded_pred = self.tokenizer.decode(item["pred"], skip_special_tokens=True)
                decoded_label = self.tokenizer.decode(item["label"], skip_special_tokens=True)
                writer.writerow({
                    "task": item["task"],
                    "pred": decoded_pred,
                    "label": decoded_label,
                    "loss": item["loss"],
                })
        self._val_buffer.clear() 

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.model.lr)