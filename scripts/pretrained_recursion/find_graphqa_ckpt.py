"""Find the GraphQA fine-tune of HRM-Text-1B (H2L3) that serves as the reference of runs d/e.

Scans the local W&B run dirs for in-distribution GraphQA runs of the hrm_text arch and lists
those with a Meta-ICL/<run id>/checkpoints/last.ckpt. Prints the newest one on the last line
(the caller writes it to $ROOT/graphqa_ckpt.txt), or exits 1 if there is none.

    python scripts/pretrained_recursion/find_graphqa_ckpt.py
"""
import ast
import glob
import json
import os
import sys

import yaml

GRL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def value(cfg, group):
    entry = cfg.get(group, {})
    entry = entry.get("value", {}) if isinstance(entry, dict) else {}
    if isinstance(entry, str):  # some runs log a group as the repr of a dict
        try:
            entry = ast.literal_eval(entry)
        except (ValueError, SyntaxError):
            return {}
    return entry if isinstance(entry, dict) else {}


def main():
    found = []
    for run_dir in sorted(glob.glob(os.path.join(GRL, "wandb", "run-*"))):
        try:
            with open(os.path.join(run_dir, "files", "config.yaml")) as f:
                cfg = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            continue
        model, dataset, trainer = value(cfg, "model"), value(cfg, "dataset"), value(cfg, "trainer")
        if not (model.get("name") == "sapientinc/HRM-Text-1B" and model.get("arch") == "hrm_text"
                and dataset.get("name") == "graphqa" and dataset.get("dataset_config") == "baseline"
                and dataset.get("test_type") == "standard" and not model.get("hrm_overrides")
                and not model.get("init_from_ckpt")):
            continue
        run_id = run_dir.rsplit("-", 1)[-1]
        ckpt = os.path.join(GRL, "Meta-ICL", run_id, "checkpoints", "last.ckpt")
        if not os.path.exists(ckpt):
            continue
        summary = {}
        try:
            with open(os.path.join(run_dir, "files", "wandb-summary.json")) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        found.append((os.path.getmtime(ckpt), run_id, ckpt, trainer.get("gradient_clip_val"),
                      summary.get("epoch"), summary.get("val/exact_match")))

    if not found:
        print("no GraphQA hrm_text (standard) run with a last.ckpt found", file=sys.stderr)
        sys.exit(1)
    for _, run_id, ckpt, clip, epoch, em in sorted(found):
        print(f"# {run_id}  clip={clip}  epoch={epoch}  val/exact_match={em}  {ckpt}", file=sys.stderr)
    newest = max(found)
    if newest[3] != 1.0:
        print(f"# WARNING: {newest[1]} used gradient_clip_val={newest[3]}; runs d/e use 1.0", file=sys.stderr)
    print(newest[2])


if __name__ == "__main__":
    main()
