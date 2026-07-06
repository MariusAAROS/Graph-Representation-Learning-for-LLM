from transformers import AutoTokenizer, AutoModelForCausalLM
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
        
        self.model = AutoModelForCausalLM.from_pretrained(self.hparams.model.name, device_map="auto")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

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
        preds = torch.argmax(logits, dim=-1)
        self.log("val/loss", loss, prog_bar=True)
    
        for i in range(len(preds)):
            self._val_buffer.append({
                "pred": preds[i],
                "label": labels[i],
                "loss": loss.item(),
            })

    def on_validation_epoch_end(self):
        if not self._val_buffer:
            return
        # Calculate accuracy
        for item in self._val_buffer:
            filtering_mask = item["label"] != -100
            item["pred"] = item["pred"][filtering_mask]
            item["label"] = item["label"][filtering_mask]
            
        # self._val_buffer = [item for item in self._val_buffer if item["label"] != -100]         
        correct = sum(1 for item in self._val_buffer if torch.equal(item["pred"], item["label"]))
        accuracy = correct / len(self._val_buffer)
        self.log("val/exact_match", accuracy, prog_bar=True)

        # Save predictions to CSV
        predictions_file = os.path.join(self._predictions_dir, f"predictions_epoch_{self.current_epoch}.csv")
        with open(predictions_file, "w", newline="") as csvfile:
            fieldnames = ["pred", "label", "loss"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in self._val_buffer:
                decoded_pred = self.tokenizer.decode(item["pred"], skip_special_tokens=True)
                decoded_label = self.tokenizer.decode(item["label"], skip_special_tokens=True)
                item["pred"] = decoded_pred
                item["label"] = decoded_label
                writer.writerow(item)
        self._val_buffer.clear() 

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.model.lr)