"""Recursion probe of an HRM-Text model (HF port): is the recurrent state still used?

Same quantities as HRM-Text-for-Graphs/scripts/probe_gain_ckpt.py, so the numbers compare
with the from-scratch size-B models:
  gain         ||f(h + d + inj) - f(h + inj)|| / ||d||,  d = eps * rms(h) * N(0, I)
               h = the module's own carried state (z_L for L, z_H for H), inj = the other one.
               ~0: the module ignores its incoming state, so extra steps carry nothing.
  prenorm_rms  RMS of the residual stream before the stack's final norm.
  rms          RMS of the module's output state.
L step 0 reads the constant z_L_init and has no gain (as in the native probe).

The HF model has no probe hooks, so this attaches forward hooks to L_module / H_module and
tracks h and inj across the H/L loop of HrmTextModel.forward. Runs one sample at a time
(no padding), in fp32.

    python scripts/pretrained_recursion/probe_gain_hf.py --source pretrained --dataset graphqa
    python scripts/pretrained_recursion/probe_gain_hf.py --source <last.ckpt> --dataset kqapro --name kqapro
    python scripts/pretrained_recursion/probe_gain_hf.py --source pretrained --init fresh --dataset graphqa
"""
import argparse
import csv
import os
import subprocess

import numpy as np
import torch
import wandb
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.datasets.loaders import BaselineDataset
from src.models.hrm_text import HRMTextICL, make_hrm_collator

MODEL_NAME = "sapientinc/HRM-Text-1B"
CONDITION = "<|object_ref_start|>"  # configs/hrm_text.yaml, model.condition
STATS = ("gain", "prenorm_rms", "rms", "fusion_err")


def rms(t):
    return t.float().pow(2).mean().sqrt()


class GainProbe:
    """Forward hooks on the L and H stacks of one HrmTextModel."""

    def __init__(self, inner, eps):
        self.inner, self.eps = inner, eps
        self.rows = []
        self._busy = False  # set while re-running a stack on the perturbed state
        self._prenorm = None
        self.handles = [
            inner.L_module.register_forward_hook(self._make_hook("L"), with_kwargs=True),
            inner.H_module.register_forward_hook(self._make_hook("H"), with_kwargs=True),
            inner.L_module.final_norm.register_forward_pre_hook(self._capture_prenorm),
            inner.H_module.final_norm.register_forward_pre_hook(self._capture_prenorm),
        ]
        self.reset()

    def reset(self):
        self.z_L = self.z_H = None
        self.step = {"L": 0, "H": 0}

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def _capture_prenorm(self, module, args):
        if not self._busy:
            self._prenorm = rms(args[0]).item()

    def _make_hook(self, role):
        @torch.no_grad()
        def hook(module, args, kwargs, output):
            if self._busy:
                return
            fused = args[0]
            if self.z_L is None:  # first L call: z_L = z_L_init, so z_H = fused - z_L_init
                self.z_L = self.inner.z_L_init.to(fused.dtype).expand_as(fused)
                self.z_H = fused - self.z_L
            h, inj = (self.z_L, self.z_H) if role == "L" else (self.z_H, self.z_L)
            idx = self.step[role]
            row = dict(role=role, idx=idx, prenorm_rms=self._prenorm, rms=rms(output).item(),
                       fusion_err=(rms(fused - h - inj) / rms(fused)).item())
            if not (role == "L" and idx == 0):
                delta = torch.randn_like(h) * (self.eps * rms(h)).to(h.dtype)
                self._busy = True
                try:
                    perturbed = module(h + delta + inj, **kwargs)
                finally:
                    self._busy = False
                row["gain"] = (rms(perturbed - output) / rms(delta).clamp_min(1e-12)).item()
            self.rows.append(row)
            self.step[role] = idx + 1
            if role == "L":
                self.z_L = output
            else:
                self.z_H = output
        return hook


def load_model(source, init, overrides):
    """source: 'pretrained' or a Lightning checkpoint of HRMTextICL. init: trained | fresh."""
    if init == "fresh":
        config = AutoConfig.from_pretrained(MODEL_NAME, **overrides)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, **overrides)
        if source != "pretrained":
            state = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
            # HRMTextICL keeps the HF model under `.model`.
            state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
            model.load_state_dict(state)
            del state
    model.config.use_cache = False
    return model.cuda().eval()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="'pretrained' or a Lightning .ckpt of HRMTextICL")
    ap.add_argument("--init", default="trained", choices=["trained", "fresh"])
    ap.add_argument("--dataset", default="graphqa", help="data/<dataset>/baseline/standard/val.json")
    ap.add_argument("--name", default=None, help="run name stem (default: dataset)")
    ap.add_argument("--H", type=int, default=None, help="H_cycles to run at (default: model config)")
    ap.add_argument("--L", type=int, default=None, help="L_cycles to run at (default: model config)")
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="c")
    ap.add_argument("--project", default=os.environ.get("PR_WANDB_PROJECT", "HRM-Pretrained-Recursion"))
    ap.add_argument("--csv", default=None, help="append per-step rows here")
    args = ap.parse_args()

    overrides = {k: v for k, v in (("H_cycles", args.H), ("L_cycles", args.L)) if v is not None}
    model = load_model(args.source, args.init, overrides)
    H, L = model.config.H_cycles, model.config.L_cycles
    name = f"{args.exp}-probe-{args.name or args.dataset}-{args.init}-H{H}L{L}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collator = make_hrm_collator(tokenizer, max_length=2048, padding_side="right", condition=CONDITION)
    dataset = BaselineDataset(f"data/{args.dataset}/baseline/standard/val.json")
    picks = np.random.default_rng(args.seed).choice(len(dataset), args.n_samples, replace=False)

    probe = GainProbe(model.model, args.eps)
    torch.manual_seed(args.seed)
    rows = []
    with torch.no_grad():
        for sample_idx in picks:
            batch = collator([dataset[int(sample_idx)]])
            token_type_ids = HRMTextICL._token_type_ids(batch["labels"], batch["attention_mask"])
            probe.reset()
            probe.rows = []
            model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(),
                  token_type_ids=token_type_ids.cuda())
            rows += [dict(sample=int(sample_idx), **r) for r in probe.rows]
    probe.remove()

    # Mean over samples per (role, step); L step 0 has no gain.
    per_step = []
    for role, idx in sorted({(r["role"], r["idx"]) for r in rows}, key=lambda k: (k[0] != "L", k[1])):
        group = [r for r in rows if r["role"] == role and r["idx"] == idx]
        step = dict(role=role, idx=idx)
        for stat in STATS:
            values = [r[stat] for r in group if r.get(stat) is not None]
            step[stat] = float(np.mean(values)) if values else float("nan")
        per_step.append(step)
    meta = dict(run=name, source=args.source, init=args.init, dataset=args.dataset,
                H=H, L=L, eps=args.eps, dtype="fp32", n_samples=args.n_samples)
    summary = {}
    for role in ("L", "H"):
        for stat in ("gain", "prenorm_rms", "rms"):
            values = [s[stat] for s in per_step if s["role"] == role and not np.isnan(s[stat])]
            summary[f"probe/{role}/{stat}_mean"] = float(np.mean(values)) if values else float("nan")
    summary["probe/fusion_err_max"] = max(s["fusion_err"] for s in per_step)

    for s in per_step:
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    print(f"{name}: gain L={summary['probe/L/gain_mean']:.4f} H={summary['probe/H/gain_mean']:.4f} "
          f"prenorm L={summary['probe/L/prenorm_rms_mean']:.1f} H={summary['probe/H/prenorm_rms_mean']:.1f} "
          f"(fusion check {summary['probe/fusion_err_max']:.1e})", flush=True)

    columns = ["role", "idx", *STATS]
    run = wandb.init(project=args.project, name=name, group=args.exp, reinit=True,
                     config={**meta, "exp": args.exp, "grl_commit": git_commit()})
    run.log({**summary, "probe/per_step": wandb.Table(columns=columns,
                                                      data=[[s[c] for c in columns] for s in per_step])})
    run.finish()

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        new_file = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[*columns, *meta])
            if new_file:
                writer.writeheader()
            writer.writerows({**s, **meta} for s in per_step)
        print(f"appended {len(per_step)} rows to {args.csv}")


if __name__ == "__main__":
    main()
