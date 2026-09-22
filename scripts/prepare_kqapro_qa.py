"""Export the KQA Pro QA splits to data/kqapro/{train,val}.json.

The official `test.json` has no answers (it is held out for the leaderboard), so
only train + val are exported; the generator re-splits train into train/val and
uses the official val as its test set.

Sources, tried in order:
  1. the HuggingFace datasets cache (`drt/kqa_pro`, config `train_val`), read
     directly from the .arrow files so no network or `trust_remote_code` is needed;
  2. an already-extracted official release directory passed via --zip_dir.

Usage:
    python -m scripts.prepare_kqapro_qa
    python -m scripts.prepare_kqapro_qa --zip_dir /path/to/unzipped/KQAPro
"""

import argparse
import glob
import json
import os

OUT_DIR = "data/kqapro"
HF_CACHE_GLOB = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "datasets", "drt___kqa_pro", "*", "*", "*", "kqa_pro-{split}.arrow",
)
OFFICIAL_URL = "https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1"


def _from_hf_cache(split):
    matches = sorted(glob.glob(HF_CACHE_GLOB.format(split=split)))
    if not matches:
        return None
    from datasets import Dataset

    return [dict(r) for r in Dataset.from_file(matches[-1])]


def _from_zip_dir(zip_dir, filename):
    path = os.path.join(zip_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip_dir", default=None,
                        help="Directory holding the unzipped official release "
                             f"({OFFICIAL_URL}).")
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split, hf_split, filename in [("train", "train", "train.json"),
                                      ("val", "validation", "val.json")]:
        records = None
        if args.zip_dir:
            records = _from_zip_dir(args.zip_dir, filename)
        if records is None:
            records = _from_hf_cache(hf_split)
        if records is None:
            raise FileNotFoundError(
                f"Could not locate the KQA Pro '{split}' split. Either populate the "
                f"HuggingFace cache with drt/kqa_pro, or download {OFFICIAL_URL}, "
                f"unzip it and pass --zip_dir."
            )
        out_path = os.path.join(args.out_dir, filename)
        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"{split}: {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
