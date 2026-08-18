from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
import json
import os

KNOWN_DATASETS = frozenset({"graphqa", "kqapro"})


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def filter_records_by_task(records, include_task=None, exclude_task=None):
    """Filter records by their "task" field.

    - include_task: keep only records whose task == include_task
    - exclude_task: drop records whose task == exclude_task
    """
    if include_task is not None:
        records = [r for r in records if r["task"] == include_task]
    if exclude_task is not None:
        records = [r for r in records if r["task"] != exclude_task]
    return records


def split_holdout_task(records, task, seed=42, val_fraction=0.5):
    """Deterministically split the held-out task's records into (val, test).

    Used for leave-one-task-out OOD evaluation: the single held-out task is
    split into a validation half (for monitoring/early stopping) and a test
    half (for final evaluation).
    """
    import random as _random
    holdout = [r for r in records if r["task"] == task]
    rng = _random.Random(seed)
    order = list(range(len(holdout)))
    rng.shuffle(order)
    n_val = int(len(order) * val_fraction)
    val_idx = set(order[:n_val])
    val = [holdout[i] for i in range(len(holdout)) if i in val_idx]
    test = [holdout[i] for i in range(len(holdout)) if i not in val_idx]
    return val, test


class MetaICLDataset(Dataset):
    def __init__(self, path: str, k=0):
        self.records = read_json(path)
        self.k = k
        self.EXAMPLE_TAG = "#### Examples"
        self.QUESTION_TAG = "#### Question"

    @classmethod
    def _from_records(cls, records, k=0):
        """Build a dataset from in-memory records instead of a file path."""
        obj = cls.__new__(cls)
        obj.records = records
        obj.k = k
        obj.EXAMPLE_TAG = "#### Examples"
        obj.QUESTION_TAG = "#### Question"
        return obj

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        ex_idx = list(RandomSampler(r["example"], 
            num_samples=self.k if self.k > 0 else len(r["example"])))
        prompt = self.EXAMPLE_TAG + "\n" + \
            "\n".join([r["example"][i] for i in ex_idx]) + \
            "\n"+ self.QUESTION_TAG + "\n" + r["question"] + r["answer"]
        return {
            # "example": [r["example"][i] for i in ex_idx],
            # "question": r["question"],
            "answer": r["answer"],
            "prompt": prompt,
            "algorithm": r["algorithm"],
            "task": r["task"],
        }
    
class BaselineDataset(Dataset):
    def __init__(self, path: str):
        self.records = read_json(path)
        self.QUESTION_TAG = "#### Question"

    @classmethod
    def _from_records(cls, records):
        """Build a dataset from in-memory records instead of a file path."""
        obj = cls.__new__(cls)
        obj.records = records
        obj.QUESTION_TAG = "#### Question"
        return obj

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        prompt = self.QUESTION_TAG + "\n" + r["question"] + r["answer"]
        return {
            # "example": [r["example"][i] for i in ex_idx],
            # "question": r["question"],
            "answer": r["answer"],
            "prompt": prompt,
            "algorithm": r["algorithm"],
            "task": r["task"],
        }

def build_graphqa_datasets(cfg, splits=("train", "val", "test"), base_dir="data/"):
    """Build datasets for the requested splits from a Hydra config.

    Encapsulates the LOTO vs standard/ood branching so that both the training
    runner and the inference runner construct datasets identically.

    - LOTO (test_type == "ood" and dataset.ood_task set): the held-out task is
      excluded from train and split 50/50 into val + test; every other task
      forms the train pool. Only graphqa has disjoint enough tasks for this.
    - Otherwise: read the pre-split {split}.json files under
      data/{name}/{dataset_config}/{test_type}/.

    Returns a dict mapping each requested split name to its Dataset.
    """
    if cfg.dataset.name not in KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}. "
                         f"Expected one of {sorted(KNOWN_DATASETS)}.")

    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset.dataset_config
    ood_task = cfg.dataset.get("ood_task", None)
    is_loto = cfg.dataset.test_type == "ood" and ood_task is not None

    def _make(records, from_file=False, path=None):
        if dataset_config == "meta-icl":
            if from_file:
                return MetaICLDataset(path, k=cfg.dataset.n_examples)
            return MetaICLDataset._from_records(records, k=cfg.dataset.n_examples)
        elif dataset_config == "baseline":
            if from_file:
                return BaselineDataset(path)
            return BaselineDataset._from_records(records)
        else:
            raise ValueError(f"Unknown dataset config: {dataset_config}")

    datasets = {}

    if is_loto:
        # Leave-one-task-out (LOTO): pool contains all tasks; hold out one.
        pool_records = []
        for split in ["train", "val", "test"]:
            pool_path = os.path.join(base_dir, dataset_name, f"{dataset_config}",
                                     "ood_pool", f"{split}.json")
            if not os.path.exists(pool_path):
                raise FileNotFoundError(
                    f"OOD pool file not found: {pool_path}. "
                    f"Generate it by running dataset_generator.py with OOD_POOL_MODE=True."
                )
            pool_records.extend(read_json(pool_path))

        available_tasks = sorted({r["task"] for r in pool_records})
        if ood_task not in available_tasks:
            raise ValueError(
                f"ood_task '{ood_task}' not found in pool. Available tasks: {available_tasks}"
            )

        val_records, test_records = split_holdout_task(pool_records, ood_task)
        if "train" in splits:
            train_records = filter_records_by_task(pool_records, exclude_task=ood_task)
            datasets["train"] = _make(train_records)
        if "val" in splits:
            datasets["val"] = _make(val_records)
        if "test" in splits:
            datasets["test"] = _make(test_records)
    else:
        for split in splits:
            current_path = os.path.join(base_dir, dataset_name, f"{dataset_config}",
                                        f"{cfg.dataset.test_type}", f"{split}.json")
            if not os.path.exists(current_path):
                raise FileNotFoundError(f"File not found: {current_path}")
            datasets[split] = _make(None, from_file=True, path=current_path)

    return datasets


def make_collator(tokenizer, max_length=1024, padding_side="right"):
    if not tokenizer.is_fast:
        raise ValueError(
            "make_collator requires a fast tokenizer "
            "(e.g. GPT2TokenizerFast) for offset mapping."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = padding_side

    def collator(batch):
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]

        labels = input_ids.clone()
        for i in range(len(prompts)):
            # The answer is the suffix of the prompt, so it starts here.
            ans_char_start = len(prompts[i]) - len(answers[i])
            ends = offsets[i, :, 1]
            # Keep any token that extends past the answer boundary. This covers
            # the first answer token even when it merges the preceding space
            # (e.g. " Yes"). Special/pad tokens have offsets (0, 0) and are
            # excluded, as are padding positions via the attention mask.
            keep = (ends > ans_char_start) & (attention_mask[i] == 1)
            labels[i][~keep] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "algorithm": [item["algorithm"] for item in batch],
            "task": [item["task"] for item in batch],
        }

    return collator


if __name__ == "__main__":
    dataset = MetaICLDataset("data/graphqa/meta-icl/standard/train.json", k=5)
    print(f"Dataset length: {len(dataset)}")
    print(f"First record: {dataset.__getitem__(0)}")
