"""Dataset smoke test: validate a config end to end without training.

Checks the invariants that break silently -- record schema, the answer being a
literal suffix of the prompt (which is how `make_collator` locates the labels),
what the label mask actually decodes back to, prompt length against
`model.max_seq_len`, and that the curriculum can be built from the record fields.

Needs a fast tokenizer but no model weights and no GPU.

    python -m scripts.smoke_test_dataset --config-name kqapro_meta_icl_1b
    python -m scripts.smoke_test_dataset --config-name baseline_1b
    python -m scripts.smoke_test_dataset --config-name meta_icl_1b -o dataset.test_type=ood
"""

import argparse
import collections
import os
import sys

import numpy as np
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from src.curriculum import build_curriculum
from src.datasets.loaders import build_graphqa_datasets, make_collator
from src.models.hrm_text import make_hrm_collator

REQUIRED_KEYS = {"question", "answer", "algorithm", "task"}
SAMPLE_SIZE = 200
BATCH_SIZE = 8

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f' -- {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--tokenizer", default=None,
                        help="Override the config's model.name (e.g. a smaller cached model).")
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Hydra override, repeatable (e.g. -o dataset.test_type=ood).")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.2"):
        cfg = compose(config_name=args.config_name, overrides=args.override)

    print(f"\nconfig={args.config_name} dataset={cfg.dataset.name} "
          f"variant={cfg.dataset.dataset_config} test_type={cfg.dataset.test_type}")

    datasets = build_graphqa_datasets(cfg, splits=("train", "val", "test"))
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset)} rows")
    train = datasets["train"]

    records = train.records
    missing = REQUIRED_KEYS - set(records[0])
    check("record schema", not missing, f"missing {sorted(missing)}" if missing else "")
    if cfg.dataset.dataset_config == "meta-icl":
        check("meta-icl records carry examples", "example" in records[0])

    rng = np.random.default_rng(0)
    idx = rng.choice(len(train), size=min(SAMPLE_SIZE, len(train)), replace=False)
    items = [train[int(i)] for i in idx]

    bad_suffix = [i for i, item in enumerate(items)
                  if not item["prompt"].endswith(item["answer"])]
    check("answer is a suffix of prompt", not bad_suffix,
          f"{len(bad_suffix)}/{len(items)} violations")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or cfg.model.name, use_fast=True)
    max_len = cfg.model.max_seq_len

    lengths = np.array([len(tokenizer(item["prompt"])["input_ids"]) for item in items])
    within = float((lengths <= max_len).mean())
    check("prompts fit max_seq_len", within >= 0.95,
          f"{100 * within:.1f}% <= {max_len} "
          f"(p50={np.percentile(lengths, 50):.0f} p95={np.percentile(lengths, 95):.0f} "
          f"max={lengths.max()})")

    # Mirror runner.py: HRM wraps the prompt in its PrefixLM envelope before
    # delegating to make_collator, which must not disturb the answer suffix.
    arch = cfg.model.get("arch", "causal_lm")
    if arch == "hrm_text":
        collator = make_hrm_collator(tokenizer, max_length=max_len, padding_side="right",
                                     condition=cfg.model.get("condition", ""))
    else:
        collator = make_collator(tokenizer, max_length=max_len, padding_side="right")
    batch = collator(items[:BATCH_SIZE])
    labelled = batch["labels"] != -100
    check(f"every example has supervised tokens (arch={arch})",
          bool(labelled.any(dim=1).all()))

    # Decoding the mask is the real test: if it returns graph text instead of the
    # answer, the model is being trained to predict the wrong span.
    mismatched = []
    for i in range(len(batch["labels"])):
        decoded = tokenizer.decode(batch["labels"][i][labelled[i]], skip_special_tokens=True)
        if decoded.strip() != items[i]["answer"].strip():
            mismatched.append((decoded.strip(), items[i]["answer"].strip()))
    check("label mask decodes to the answer", not mismatched,
          f"e.g. got {mismatched[0][0]!r} want {mismatched[0][1]!r}" if mismatched else "")

    check("batch carries task and algorithm",
          len(batch["task"]) == BATCH_SIZE and len(batch["algorithm"]) == BATCH_SIZE)

    features = list(cfg.get("curriculum", {}).get("graph_features", []))
    try:
        sampler, _ = build_curriculum(cfg, train)
        check("curriculum builds", True,
              f"features={features} total_steps={sampler.total_steps}")
    except Exception as exc:  # noqa: BLE001 - surfacing any failure is the point
        check("curriculum builds", False, f"{type(exc).__name__}: {exc}")

    print("\n  tasks:      ", dict(collections.Counter(r["task"] for r in records)))
    print("  algorithms: ", dict(collections.Counter(r["algorithm"] for r in records)))

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'All checks passed.'}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
