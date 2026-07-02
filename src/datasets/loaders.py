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
    def __init__(self, path: str, k=None):
        self.records = read_json(path)
        self.k = k if k is not None else 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "examples": ...,
            "question": r["question"],
            "answer": r["answer"],
        }
    
if __name__ == "__main__":
    dataset = MetaICLDataset("data/graphqa/train.json")
    print(f"Dataset length: {len(dataset)}")
    print(f"First record: {dataset.records[0]}")
