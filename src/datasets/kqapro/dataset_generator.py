"""Generate GraphQA-schema training files from KQA Pro.

Emits `data/kqapro/{MODEL_TYPE}/standard/{train,val,test}.json` with the same record
schema the GraphQA files use, so `MetaICLDataset` / `BaselineDataset` consume them
unchanged. Each question is paired with a subgraph carved out of `kb.json`; the
seeding strategy is recorded in `algorithm` so a single run yields a `gold` vs
`retrieved` comparison in the prediction CSVs.

Run from the repository root, once per MODEL_TYPE:
    python -m src.datasets.kqapro.dataset_generator
"""

import json
import os
import random

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datasets.kqapro.kb_index import KBIndex
from src.datasets.kqapro.question_types import derive_task, program_length

KB_PATH = "data/kqapro/kb.json"
QA_DIR = "data/kqapro"
OUTPUT_DIR = "data/kqapro/"
MODEL_TYPE = "meta-icl"  # meta-icl, baseline

# (seeding strategy, hops) -> one `algorithm` value each, both written to the same files.
STRATEGIES = [("gold", 1), ("retrieved", 2)]
MAX_TRIPLES = 60      # ~1100 tokens of KB text, leaving room for 5-shot prompts at 4096
MAX_NODES = 128       # caps the k-hop expansion before triples are selected
N_TRAIN_POOL = 4000   # questions sampled from the official train split
N_TEST = 600          # questions sampled from the official val split (test.json has no answers)
VAL_FRACTION = 0.1
N_EXAMPLES_PER_SAMPLE = 32
SEED = 42

# Counts and yes/no verdicts are computed from the graph rather than stated in it,
# so they are exempt from the check that a gold subgraph contains its answer.
DERIVED_TASKS = frozenset({"Count", "Verify"})

SPLITS = ["train", "val", "test"]


def build_rows(records, kb, rng, desc):
    rows, dropped = [], 0
    for record in tqdm(records, desc=desc):
        task = derive_task(record["program"])
        nsteps = program_length(record["program"])
        for strategy, hops in STRATEGIES:
            built = kb.build_subgraph(record, strategy, hops,
                                      max_triples=MAX_TRIPLES,
                                      max_nodes=MAX_NODES, rng=rng)
            if built is None:
                dropped += 1
                continue
            text, nnodes, nedges = built
            # An unanswerable gold example would only teach the model to guess;
            # under `retrieved`, missing evidence is the phenomenon being measured.
            if (strategy == "gold" and task not in DERIVED_TASKS
                    and str(record["answer"]) not in text):
                dropped += 1
                continue
            rows.append({
                "algorithm": f"{strategy}_{hops}hop",
                "task": task,
                "nnodes": str(nnodes),
                "nedges": str(nedges),
                "nsteps": str(nsteps),
                "question": f"{text}Q: {record['question']}\nA: ",
                "answer": f"{record['answer']}.",
            })
    print(f"  {desc}: kept {len(rows)}, dropped {dropped}")
    return rows


def build_example_bank(records):
    """Few-shot pools keyed by task, drawn from the train questions only.

    Unlike GraphQA, the examples carry no subgraph: five serialized KBs would not
    fit alongside the query's own subgraph in the context window.
    """
    bank = {}
    for record in records:
        task = derive_task(record["program"])
        bank.setdefault(task, []).append(
            f"Question:\n{record['question']}\nA: {record['answer']}."
        )
    return bank


def attach_examples(rows, bank, rng):
    for row in rows:
        pool = bank[row["task"]]
        row["example"] = [rng.choice(pool) for _ in range(N_EXAMPLES_PER_SAMPLE)]


def main():
    rng = random.Random(SEED)
    print(f"Loading KB from {KB_PATH} ...")
    kb = KBIndex(KB_PATH)

    with open(os.path.join(QA_DIR, "train.json")) as f:
        official_train = json.load(f)
    with open(os.path.join(QA_DIR, "val.json")) as f:
        official_val = json.load(f)

    pool = rng.sample(official_train, min(N_TRAIN_POOL, len(official_train)))
    train_records, val_records = train_test_split(
        pool, test_size=VAL_FRACTION, random_state=SEED)
    test_records = rng.sample(official_val, min(N_TEST, len(official_val)))

    split_records = {"train": train_records, "val": val_records, "test": test_records}
    split_rows = {split: build_rows(records, kb, rng, desc=f"subgraphs [{split}]")
                  for split, records in split_records.items()}

    if MODEL_TYPE == "meta-icl":
        bank = build_example_bank(train_records)
        for rows in split_rows.values():
            attach_examples(rows, bank, rng)
    elif MODEL_TYPE != "baseline":
        raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'meta-icl' or 'baseline'.")

    for split in SPLITS:
        rows = split_rows[split]
        data = [{"id": f"{split}_{i}", **row} for i, row in enumerate(rows)]
        output_file = os.path.join(OUTPUT_DIR, MODEL_TYPE, "standard", f"{split}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"{split}: {len(data)} rows -> {output_file}")


if __name__ == "__main__":
    main()
