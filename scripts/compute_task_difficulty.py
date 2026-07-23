"""Precompute per-task OOD difficulty from local wandb summaries.

Scans ``wandb/*/files/wandb-summary.json`` for leave-one-task-out runs and
writes a task -> difficulty (0..1) JSON cache consumed by the curriculum. Run
once after new OOD runs, or let the training runner compute it lazily.

Usage:
    python -m scripts.compute_task_difficulty \
        --wandb-dir wandb \
        --out configs/curriculum/task_difficulty.json \
        --metric gap
"""

import argparse
import json

from src.curriculum.difficulty import (
    compute_task_difficulty,
    parse_ood_task_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-dir", default="wandb")
    parser.add_argument("--out", default="configs/curriculum/task_difficulty.json")
    parser.add_argument("--metric", choices=["gap", "raw"], default="gap")
    parser.add_argument(
        "--run-name-prefix",
        default="meta-icl-ood-",
        help="Only use LOTO runs whose reconstructed name starts with this "
        "prefix. Pass an empty string to use all runs.",
    )
    args = parser.parse_args()

    prefix = args.run_name_prefix or None
    metrics = parse_ood_task_metrics(args.wandb_dir, run_name_prefix=prefix)
    if not metrics:
        print(f"[warn] no LOTO OOD runs found under {args.wandb_dir!r}")
    difficulty = compute_task_difficulty(metrics, metric=args.metric)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(difficulty, f, indent=2, sort_keys=True)

    print(f"[ok] wrote {len(difficulty)} task difficulties -> {args.out}")
    for task in sorted(difficulty, key=difficulty.get):
        m = metrics.get(task, {})
        print(
            f"  {task:20s} difficulty={difficulty[task]:.3f} "
            f"(exact_match={m.get('exact_match', float('nan')):.3f}, "
            f"baseline={m.get('baseline', float('nan')):.3f}, n={m.get('n', 0)})"
        )


if __name__ == "__main__":
    main()
