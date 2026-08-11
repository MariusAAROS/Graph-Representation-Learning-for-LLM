"""Pre-flight tokenizer check for every backbone in configs/model/.

Two things silently break when swapping the backbone, and neither raises:
  1. make_collator locates the answer by character offset, so a tokenizer whose
     offsets do not line up would train on the wrong span.
  2. truncation_side="left" drops few-shot examples once a prompt exceeds
     model.max_seq_len, and tokenizers differ a lot on graph text.

Run on a Jean Zay login node after scripts/jeanzay/fetch_models.sh:
    python scripts/check_tokenizers.py
    python scripts/check_tokenizers.py --config-name meta_icl_1b --models qwen2_5_3b
"""
import argparse
import glob
import os
import sys

from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.loaders import build_graphqa_datasets, make_collator  # noqa: E402

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)


def check(model_file: str, config_name: str, n_samples: int) -> bool:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.2"):
        cfg = compose(config_name=config_name, overrides=[f"model={model_file}"])

    name = cfg.model.name
    print(f"\n=== {model_file}  ({name}) ===")

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    ok = True

    if not tok.is_fast:
        print("  FAIL: not a fast tokenizer; make_collator needs offset mapping")
        return False
    if tok.eos_token is None and tok.pad_token is None:
        print("  FAIL: neither pad_token nor eos_token; padding would crash")
        return False

    dataset = build_graphqa_datasets(cfg, splits=("train",))["train"]
    items = [dataset[i] for i in range(min(n_samples, len(dataset)))]

    # 1. Answer span must survive tokenization.
    collator = make_collator(tok, max_length=cfg.model.max_seq_len, padding_side="right")
    mismatches = 0
    for start in range(0, len(items), 8):
        batch = items[start:start + 8]
        out = collator(batch)
        for i, item in enumerate(batch):
            kept = out["labels"][i][out["labels"][i] != -100]
            decoded = tok.decode(kept, skip_special_tokens=True).strip()
            if decoded != item["answer"].strip():
                mismatches += 1
                if mismatches <= 3:
                    print(f"  answer mismatch: got {decoded!r} want {item['answer'].strip()!r}")
    if mismatches:
        print(f"  FAIL: {mismatches}/{len(items)} answers not recoverable from labels")
        ok = False
    else:
        print(f"  ok: answer span recovered for all {len(items)} samples")

    # 2. Prompts longer than max_seq_len get left-truncated, losing examples.
    lengths = sorted(len(tok(item["prompt"])["input_ids"]) for item in items)
    limit = cfg.model.max_seq_len
    over = sum(1 for n in lengths if n > limit)
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[int(len(lengths) * 0.95)]
    print(f"  prompt tokens: p50={p50} p95={p95} max={lengths[-1]} (max_seq_len={limit})")
    if over:
        print(f"  WARN: {over}/{len(lengths)} prompts truncated; raise model.max_seq_len")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="baseline_1b")
    parser.add_argument("--models", nargs="*", help="default: every configs/model/*.yaml")
    parser.add_argument("--n-samples", type=int, default=64)
    args = parser.parse_args()

    models = args.models or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(CONFIG_DIR, "model", "*.yaml"))
    )

    failed = [m for m in models if not check(m, args.config_name, args.n_samples)]
    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(models)} tokenizers OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
