import torch
from torch import nn

from src.datasets.graphqa.structure import GraphStructureFeaturizer
from src.datasets.loaders import make_collator
from src.models.meta_icl import MetaICL


def make_hrm_collator(tokenizer, max_length=1024, padding_side="right", condition="",
                      structure_fn=None):
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

    ``structure_fn`` is forwarded to :func:`make_collator`. It sees the wrapped
    prompt, so its spans need no offset correction.
    """
    prefix = f"<|im_start|>{condition}"
    base = make_collator(tokenizer, max_length=max_length, padding_side=padding_side,
                         structure_fn=structure_fn)

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

    def _model_inputs(self, batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": self._token_type_ids(batch["labels"], batch["attention_mask"]),
        }

    def training_step(self, batch):
        outputs = self.model(**self._model_inputs(batch), labels=batch["labels"])
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch):
        labels = batch["labels"]
        outputs = self.model(**self._model_inputs(batch), labels=labels)
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


def make_structure_fn(cfg, train):
    """Featurizer for ``arch: hrm_graph``, or None for any other arch.

    Training draws a fresh node permutation each time an example is seen.
    Evaluation seeds it from ``(structure.seed, prompt)``, so val/test are
    reproducible.
    """
    if cfg.model.get("arch") != "hrm_graph":
        return None
    s = cfg.structure
    return GraphStructureFeaturizer(
        num_node_rows=s.num_node_rows,
        num_wl_rows=s.num_wl_rows,
        node_mode=s.node_mode,
        wl_mode=s.wl_mode,
        deterministic=not train,
        seed=s.seed,
    )


class HRMGraphICL(HRMTextICL):
    """HRM-Text with gated structural embeddings on node-mention tokens.

    For token t with node row n_t and WL row w_t (-1 = none):

        z_H^0_t = s * ( W_tok[x_t] + g_node * E_node[n_t] + g_wl * E_wl[w_t] )

    where ``s`` is HRM's ``embedding_scale``, applied inside the model to
    ``inputs_embeds``. The rows of ``E`` are drawn from N(0, sigma^2) with sigma
    equal to the RMS of ``W_tok``, and the gates start at 0, so step 0 is
    exactly the pretrained model. They cannot both start at zero, because each
    one's gradient is proportional to the other. With
    ``structure.train_codes: false`` the tables stay at their random init and
    only the gates learn (the A2-frozen ablation).
    """

    def __init__(self, config):
        super().__init__(config)
        s = self.hparams.structure
        token_embed = self.model.get_input_embeddings().weight
        hidden = token_embed.shape[1]
        sigma = token_embed.detach().float().pow(2).mean().sqrt().item()
        self.node_embed = self._code_table(s.node, s.num_node_rows, hidden, sigma, s.train_codes)
        self.wl_embed = self._code_table(s.wl, s.num_wl_rows, hidden, sigma, s.train_codes)
        self.node_gate = nn.Parameter(torch.zeros(())) if s.node else None
        self.wl_gate = nn.Parameter(torch.zeros(())) if s.wl else None

    @staticmethod
    def _code_table(enabled, rows, hidden, sigma, trainable):
        if not enabled:
            return None
        table = nn.Embedding(rows, hidden)
        nn.init.normal_(table.weight, mean=0.0, std=sigma)
        table.weight.requires_grad_(bool(trainable))
        return table

    def _structure_terms(self):
        return [(key, table, gate) for key, table, gate in (
            ("node_ids", self.node_embed, self.node_gate),
            ("wl_ids", self.wl_embed, self.wl_gate),
        ) if table is not None]

    def _model_inputs(self, batch):
        inputs = super()._model_inputs(batch)
        embeds = self.model.get_input_embeddings()(inputs.pop("input_ids"))
        extra = torch.zeros_like(embeds, dtype=torch.float32)
        for key, table, gate in self._structure_terms():
            ids = batch[key]
            mask = (ids >= 0).unsqueeze(-1)
            extra = extra + gate * table(ids.clamp(min=0)).float() * mask
        inputs["inputs_embeds"] = embeds + extra.to(embeds.dtype)
        return inputs

    def training_step(self, batch):
        loss = super().training_step(batch)
        if self.node_gate is not None:
            self.log("struct/node_gate", self.node_gate.detach())
        if self.wl_gate is not None:
            self.log("struct/wl_gate", self.wl_gate.detach())
        return loss

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        params = [p for _, table, gate in self._structure_terms()
                  for p in (table.weight, gate) if p.requires_grad]
        if params:
            optimizer.add_param_group({"params": params, "lr": self.hparams.structure.lr})
        return optimizer
