import torch

from src.datasets.loaders import make_collator
from src.models.meta_icl import MetaICL


def make_hrm_collator(tokenizer, max_length=1024, padding_side="right", condition=""):
    """Collator for the HRM-Text PrefixLM checkpoint.

    Wraps each training string in HRM's pretraining envelope
    ``<|im_start|>{condition}{question}<|im_end|>{answer}`` and then delegates to
    the shared :func:`make_collator` for tokenization and answer-only label
    masking. Because ``answer`` is the exact suffix of ``prompt`` (see the
    dataset classes in ``src/datasets/loaders.py``), re-wrapping the question
    part keeps ``answer`` as the suffix, so the char-offset label boundary logic
    in the base collator remains correct.

    ``condition`` is the composite prefix tag string (e.g. ``<|object_ref_start|>``
    for the ``direct`` mode). Pass ``""`` to use no condition.
    """
    prefix = f"<|im_start|>{condition}"
    base = make_collator(tokenizer, max_length=max_length, padding_side=padding_side)

    def _wrap(item):
        prompt = item["prompt"]
        answer = item["answer"]
        question = prompt[: len(prompt) - len(answer)]
        new_prompt = f"{prefix}{question}<|im_end|>{answer}"
        return {**item, "prompt": new_prompt}

    def collator(batch):
        return base([_wrap(item) for item in batch])

    return collator


class HRMTextICL(MetaICL):
    """MetaICL variant for the HRM-Text PrefixLM checkpoint.

    HRM-Text was pretrained with a PrefixLM objective: prompt tokens attend
    bidirectionally, response tokens causally. In the Transformers port this
    mask is controlled by ``token_type_ids`` (1 = prefix). We derive it from the
    label mask -- prompt/prefix tokens carry label -100 -- so the shared collator
    stays model-agnostic and GPT-2 (which reads ``token_type_ids`` as segment
    embeddings) is never handed one.
    """

    @staticmethod
    def _token_type_ids(labels, attention_mask):
        # Prefix = real (non-pad) tokens that are not part of the answer.
        # Answer tokens (label != -100) stay causal.
        return ((labels == -100) & (attention_mask == 1)).long()

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = self._token_type_ids(labels, attention_mask)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = self._token_type_ids(labels, attention_mask)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
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
