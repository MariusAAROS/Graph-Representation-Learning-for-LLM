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

class MetaICLDataset(Dataset):
    def __init__(self, path: str, k=0):
        self.records = read_json(path)
        self.k = k
        self.EXAMPLE_TAG = "#### Examples"
        self.QUESTION_TAG = "#### Question"

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
