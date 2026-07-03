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
            "example": [r["example"][i] for i in ex_idx],
            "question": r["question"],
            "answer": r["answer"],
            "prompt": prompt
        }

def make_meta_icl_collator(tokenizer):
    def collator(batch):
        pass
    return collator


if __name__ == "__main__":
    dataset = MetaICLDataset("data/graphqa/meta-icl/standard/train.json", k=5)
    print(f"Dataset length: {len(dataset)}")
    print(f"First record: {dataset.__getitem__(0)}")
