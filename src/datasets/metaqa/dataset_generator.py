"""Generate GraphQA-schema training files from MetaQA.

Emits `data/metaqa/{baseline,meta-icl}/standard/{train,val,test}.json` with the same
record schema the GraphQA and KQA Pro files use, so `MetaICLDataset` /
`BaselineDataset` consume them unchanged. Each question is paired with a subgraph
carved out of `kb.txt`; the seeding strategy is recorded in `algorithm` so a single
run yields a `gold` vs `retrieved` comparison in the prediction CSVs.

Only single-answer questions are kept, which makes `val/exact_match` equal to the
Hits@1 that MetaQA is normally scored with.

Run from the repository root:
    python -m src.datasets.metaqa.dataset_generator
"""

import json
import os
import random
import re

from tqdm import tqdm

from src.datasets.metaqa.kb_index import KBIndex
from src.datasets.metaqa.question_types import derive_task, qtype_chain

KB_PATH = "data/metaqa/kb.txt"
DATA_DIR = "data/metaqa"
OUTPUT_DIR = "data/metaqa"
MODEL_TYPES = ["baseline", "meta-icl"]

HOPS = [1, 2, 3]
STRATEGIES = ["gold", "retrieved"]
MAX_TRIPLES = 80      # MetaQA triples are short (~12 tokens), so 80 fits in ~1000
MAX_NODES = 200       # caps the retrieved ball before triples are selected
N_TRAIN = 4000        # split evenly across the three hop levels
N_VAL = 400
N_TEST = 600
N_EXAMPLES_PER_SAMPLE = 32
SEED = 42

# MetaQA ships an official three-way split; unlike KQA Pro every split has answers.
SPLIT_FILES = {"train": "qa_train", "val": "qa_dev", "test": "qa_test"}

TOPIC_RE = re.compile(r"\[(.+?)\]")


def load_records(hops, split, n_samples, rng):
    """Read one hop level's QA file, keep single-answer rows, subsample."""
    qa_path = os.path.join(DATA_DIR, f"{hops}-hop", "vanilla", f"{SPLIT_FILES[split]}.txt")
    qtype_path = os.path.join(DATA_DIR, f"{hops}-hop", f"{SPLIT_FILES[split]}_qtype.txt")
    with open(qa_path) as f:
        lines = f.read().splitlines()
    with open(qtype_path) as f:
        qtypes = f.read().splitlines()

    records = []
    for line, qtype in zip(lines, qtypes):
        question, _, answers = line.partition("\t")
        answers = answers.split("|")
        if len(answers) != 1:
            continue
        topic = TOPIC_RE.search(question)
        if topic is None:
            continue
        records.append({
            # Brackets are stripped so `retrieved` is not handed the topic entity for
            # free; `gold` uses the annotated span, which the prompt never shows.
            "question": TOPIC_RE.sub(r"\1", question),
            "topic": topic.group(1),
            "answer": answers[0],
            "chain": qtype_chain(qtype),
            "nsteps": hops,
            "task": derive_task(hops),
        })
    return rng.sample(records, min(n_samples, len(records)))


def build_rows(records, kb, rng, desc):
    rows, dropped, covered = [], 0, {strategy: [0, 0] for strategy in STRATEGIES}
    for record in tqdm(records, desc=desc):
        for strategy in STRATEGIES:
            built = kb.build_subgraph(record, strategy, max_triples=MAX_TRIPLES,
                                      max_nodes=MAX_NODES, rng=rng)
            if built is None:
                dropped += 1
                continue
            text, nnodes, nedges = built
            covered[strategy][1] += 1
            covered[strategy][0] += record["answer"] in text
            # An unanswerable gold example would only teach the model to guess;
            # under `retrieved`, missing evidence is the phenomenon being measured.
            if strategy == "gold" and record["answer"] not in text:
                dropped += 1
                continue
            rows.append({
                "algorithm": strategy,
                "task": record["task"],
                "nnodes": str(nnodes),
                "nedges": str(nedges),
                "nsteps": str(record["nsteps"]),
                "question": f"{text}Q: {record['question']}\nA: ",
                "answer": f"{record['answer']}.",
            })
    rates = " | ".join(f"{s} answer-in-graph {100 * hit / max(1, total):.1f}%"
                       for s, (hit, total) in covered.items())
    print(f"  {desc}: kept {len(rows)}, dropped {dropped} | {rates}")
    return rows


def build_example_bank(records):
    """Few-shot pools keyed by task, drawn from the train questions only.

    As with KQA Pro the examples carry no subgraph: several serialized KBs would not
    fit alongside the query's own.
    """
    bank = {}
    for record in records:
        bank.setdefault(record["task"], []).append(
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

    sizes = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}
    split_records, split_rows = {}, {}
    for split, total in sizes.items():
        records = []
        for hops in HOPS:
            records.extend(load_records(hops, split, total // len(HOPS), rng))
        rng.shuffle(records)
        split_records[split] = records
        split_rows[split] = build_rows(records, kb, rng, desc=f"subgraphs [{split}]")

    bank = build_example_bank(split_records["train"])
    for model_type in MODEL_TYPES:
        for split, rows in split_rows.items():
            rows = [dict(row) for row in rows]
            if model_type == "meta-icl":
                attach_examples(rows, bank, rng)
            data = [{"id": f"{split}_{i}", **row} for i, row in enumerate(rows)]
            output_file = os.path.join(OUTPUT_DIR, model_type, "standard", f"{split}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"{model_type}/{split}: {len(data)} rows -> {output_file}")


if __name__ == "__main__":
    main()
