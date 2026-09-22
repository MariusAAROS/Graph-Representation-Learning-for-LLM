from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import os, csv, re
from datetime import datetime
import pytorch_lightning as pl
import torch


def normalize_answer(text):
    """Lowercase and drop articles, punctuation and redundant whitespace.

    GraphQA answers are templated, but KQA Pro answers are free-form KB values, so
    a trailing period or a leading "the" should not count as a wrong answer.
    """
    text = re.sub(r"[.,;:!?\"'`()\[\]]+", " ", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


SET_METRICS = ("set_precision", "set_recall", "set_f1", "hits_at_1")

# Matches ANSWER_SEP in the MetaQA generator; a comma would be ambiguous because
# entity names contain commas.
ANSWER_SEP = "|"


def answer_list(text):
    """Split a separator-joined answer string into normalized, non-empty elements.

    Splitting must precede `normalize_answer`, which collapses punctuation.
    """
    parts = (normalize_answer(part) for part in text.strip().rstrip(".").split(ANSWER_SEP))
    return [part for part in parts if part]


def set_scores(pred, label):
    """Precision / recall / F1 / hits@1 between two separator-joined answer lists.

    Degenerates to exact_match_norm on single-answer datasets, so it is safe to log
    everywhere.
    """
    gold, got = set(answer_list(label)), answer_list(pred)
    if not gold:
        return 0.0, 0.0, 0.0, 0.0
    hit = len(gold & set(got))
    precision = hit / len(got) if got else 0.0
    recall = hit / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return precision, recall, f1, float(bool(got) and got[0] in gold)


class MetaICL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Warning: No pad_token found for {self.hparams.model.name}, setting pad_token to eos_token ({self.tokenizer.eos_token})")
        self.tokenizer.padding_side = "left"
        
        load_dtype = getattr(self.hparams.model, "load_dtype", None)
        torch_dtype = None
        if load_dtype not in (None, "", "null", "float32", "fp32"):
            torch_dtype = getattr(torch, load_dtype)

        load_kwargs = {}
        attn_impl = getattr(self.hparams.model, "attn_implementation", None)
        if attn_impl not in (None, "", "null"):
            load_kwargs["attn_implementation"] = attn_impl

        if torch_dtype is not None:
            # transformers >=5 renamed `torch_dtype` to `dtype`; try the modern
            # kwarg first and fall back for older versions.
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hparams.model.name, dtype=torch_dtype, **load_kwargs
                )
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hparams.model.name, torch_dtype=torch_dtype, **load_kwargs
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hparams.model.name, **load_kwargs
            )
        self.model.train()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # KV cache is useless during teacher-forced training and conflicts with
        # gradient checkpointing (HF disables it with a warning); turn it off so
        # the HRM port cannot silently retain cache activations.
        self.model.config.use_cache = False
        if getattr(self.hparams.model, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            # The HRM-Text port may not honour enable(); verify it actually took
            # effect so we don't silently pay full-activation memory.
            if not getattr(self.model, "is_gradient_checkpointing", False):
                print(
                    "Warning: gradient_checkpointing requested but "
                    f"{self.hparams.model.name} did not enable it "
                    "(is_gradient_checkpointing is False); activations will not "
                    "be checkpointed."
                )

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
        # Per-task majority-class random baseline, cached after first epoch
        # since the validation set is fixed across epochs.
        self._task_baselines: dict = {}

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

        # Decode labels once (reused for the baseline and the CSV dump).
        for item in self._val_buffer:
            item["decoded_label"] = self.tokenizer.decode(
                item["label"], skip_special_tokens=True
            )
            item["decoded_pred"] = self.tokenizer.decode(
                item["pred"], skip_special_tokens=True
            )

        # Token-id equality above is tokenizer-dependent ("23" is one token for
        # some backbones and two for others), so it is not comparable across
        # models; this string-level variant is.
        str_correct = sum(
            1 for item in self._val_buffer
            if item["decoded_pred"].strip() == item["decoded_label"].strip()
        )
        self.log("val/exact_match_str", str_correct / total)

        norm_correct = sum(
            1 for item in self._val_buffer
            if normalize_answer(item["decoded_pred"]) == normalize_answer(item["decoded_label"])
        )
        self.log("val/exact_match_norm", norm_correct / total)

        # Exact match over the whole string is all-or-nothing on multi-answer targets
        # (MetaQA variants), where partial credit is the quantity of interest.
        for item in self._val_buffer:
            scores = set_scores(item["decoded_pred"], item["decoded_label"])
            item.update(zip(SET_METRICS, scores))
        set_means = {
            key: sum(item[key] for item in self._val_buffer) / total
            for key in SET_METRICS
        }
        for key, value in set_means.items():
            self.log(f"val/{key}", value)

        # Per-task majority-class random baseline. For each task, this is the
        # frequency of the most common answer, i.e. the accuracy of always
        # predicting that answer. It gives a data-driven chance level that works
        # for every task type: ~0.5 for balanced binary tasks (e.g. cycle_check)
        # and a meaningful floor for numeric tasks (e.g. maximum_flow). Computed
        # once and cached, since the validation set is fixed across epochs.
        # Skip during the sanity-check pass: it only runs on a couple of batches,
        # so the per-task label counts are degenerate (often a single answer,
        # giving a bogus baseline of 1.0) and would be cached permanently.
        if not self.trainer.sanity_checking and not self._task_baselines:
            task_label_counts: dict = {}
            for item in self._val_buffer:
                task = item["task"] if item["task"] is not None else "unknown"
                counts = task_label_counts.setdefault(task, {})
                label = item["decoded_label"]
                counts[label] = counts.get(label, 0) + 1
            for task, counts in task_label_counts.items():
                self._task_baselines[task] = max(counts.values()) / sum(counts.values())

        # Per-task exact-match accuracy (some tasks are harder than others)
        task_correct: dict = {}
        task_str_correct: dict = {}
        task_norm_correct: dict = {}
        task_set_sums: dict = {}
        task_total: dict = {}
        for item in self._val_buffer:
            task = item["task"] if item["task"] is not None else "unknown"
            task_total[task] = task_total.get(task, 0) + 1
            if torch.equal(item["pred"], item["label"]):
                task_correct[task] = task_correct.get(task, 0) + 1
            if item["decoded_pred"].strip() == item["decoded_label"].strip():
                task_str_correct[task] = task_str_correct.get(task, 0) + 1
            if normalize_answer(item["decoded_pred"]) == normalize_answer(item["decoded_label"]):
                task_norm_correct[task] = task_norm_correct.get(task, 0) + 1
            sums = task_set_sums.setdefault(task, dict.fromkeys(SET_METRICS, 0.0))
            for key in SET_METRICS:
                sums[key] += item[key]
        for task in sorted(task_total):
            task_acc = task_correct.get(task, 0) / task_total[task]
            task_str_acc = task_str_correct.get(task, 0) / task_total[task]
            task_norm_acc = task_norm_correct.get(task, 0) / task_total[task]
            task_set_means = {key: value / task_total[task]
                              for key, value in task_set_sums[task].items()}
            baseline = self._task_baselines.get(task, 0.0)
            self.log(f"val/exact_match_{task}", task_acc)
            self.log(f"val/exact_match_str_{task}", task_str_acc)
            self.log(f"val/exact_match_norm_{task}", task_norm_acc)
            for key, value in task_set_means.items():
                self.log(f"val/{key}_{task}", value)
            self.log(f"val/random_baseline_{task}", baseline)
            print(
                f"[val] epoch {self.current_epoch} | task={task} | "
                f"exact_match={task_acc:.4f} "
                f"({task_correct.get(task, 0)}/{task_total[task]}) | "
                f"exact_match_str={task_str_acc:.4f} | "
                f"exact_match_norm={task_norm_acc:.4f} | "
                f"set_f1={task_set_means['set_f1']:.4f} | "
                f"hits@1={task_set_means['hits_at_1']:.4f} | "
                f"baseline={baseline:.4f}"
            )
        print(
            f"[val] epoch {self.current_epoch} | task=ALL | "
            f"exact_match={accuracy:.4f} ({correct}/{total}) | "
            f"exact_match_str={str_correct / total:.4f} | "
            f"exact_match_norm={norm_correct / total:.4f} | "
            f"set_f1={set_means['set_f1']:.4f} | "
            f"hits@1={set_means['hits_at_1']:.4f}"
        )

        # Save predictions to CSV
        predictions_file = os.path.join(self._predictions_dir, f"predictions_epoch_{self.current_epoch}.csv")
        with open(predictions_file, "w", newline="") as csvfile:
            fieldnames = ["task", "pred", "label", "set_f1", "loss"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in self._val_buffer:
                writer.writerow({
                    "task": item["task"],
                    "pred": item["decoded_pred"],
                    "label": item["decoded_label"],
                    "set_f1": item["set_f1"],
                    "loss": item["loss"],
                })
        self._val_buffer.clear() 

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = self.hparams.model.lr
        name = getattr(self.hparams.model, "optimizer", "adamw")

        if name in ("adamw_8bit", "paged_adamw_8bit"):
            try:
                import bitsandbytes as bnb
            except ImportError:
                print(
                    f"Warning: optimizer '{name}' requested but bitsandbytes is "
                    "not installed; falling back to torch.optim.AdamW."
                )
                return torch.optim.AdamW(params, lr=lr)

            optim_cls = (
                bnb.optim.PagedAdamW8bit
                if name == "paged_adamw_8bit"
                else bnb.optim.AdamW8bit
            )
            # 8-bit optimizer state on embeddings is a common source of
            # instability; keep those layers in 32-bit optimizer state. The
            # override is consulted lazily at the first step, so registering it
            # before creating the optimizer is sufficient.
            manager = bnb.optim.GlobalOptimManager.get_instance()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
            return optim_cls(params, lr=lr)

        return torch.optim.AdamW(params, lr=lr)
