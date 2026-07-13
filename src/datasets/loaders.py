from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
import json

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
