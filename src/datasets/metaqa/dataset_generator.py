"""Generate GraphQA-schema training files from MetaQA.

Emits `data/metaqa/{baseline,meta-icl}/standard/{train,val,test}.json` with the same
record schema the GraphQA and KQA Pro files use, so `MetaICLDataset` /
`BaselineDataset` consume them unchanged. Each question is paired with a subgraph
carved out of `kb.txt`; the seeding strategy is recorded in `algorithm` so a single
run yields a `gold` vs `retrieved` comparison in the prediction CSVs.

The answer-set size is a parameter. At the default of exactly one answer,
`val/exact_match` is the Hits@1 that MetaQA is normally scored with; with a wider
window the target becomes a sorted, `ANSWER_SEP`-joined list and the set metrics
logged by `MetaICL` (`val/set_f1`, `val/hits_at_1`) are the ones to read.

Run from the repository root:
    python -m src.datasets.metaqa.dataset_generator
    python -m src.datasets.metaqa.dataset_generator --answer-len 5 --interv 2 \\
        --strategies gold --output-dir data/metaqa-multi
"""

import argparse
import json
import os
import random
import re

from tqdm import tqdm

from src.datasets.metaqa.kb_index import KBIndex
from src.datasets.metaqa.question_types import derive_task, qtype_chain

KB_PATH = "data/metaqa/kb.txt"
DATA_DIR = "data/metaqa"
OUTPUT_DIR = "data/metaqa-gold"
MODEL_TYPES = ["baseline", "meta-icl"]

HOPS = [1, 2, 3]
# STRATEGIES = ["gold", "retrieved"]
STRATEGIES = ["gold"]
MAX_TRIPLES = 80      # MetaQA triples are short (~12 tokens), so 80 fits in ~1000
MAX_NODES = 200       # caps the retrieved ball before triples are selected
N_TRAIN = 4000        # split evenly across the three hop levels
N_VAL = 400
N_TEST = 600
N_EXAMPLES_PER_SAMPLE = 32
SEED = 42
ANSWER_LEN = 1            # target answer-set size
INTERV_ANSWER_LEN = 0     # half-width of the accepted window around ANSWER_LEN
# Answers are sorted before joining: the serialized subgraph is shuffled, so any
# order derived from it would be unlearnable. ", " is not usable as the separator --
# 244 KB entities contain a comma ("Rome, Open City") -- but "|" is the field
# separator of kb.txt, so no entity name can contain it.
ANSWER_SEP = " | "

# MetaQA ships an official three-way split; unlike KQA Pro every split has answers.
SPLIT_FILES = {"train": "qa_train", "val": "qa_dev", "test": "qa_test"}

TOPIC_RE = re.compile(r"\[(.+?)\]")


def load_records(hops, split, n_samples, rng,
                 answer_len=ANSWER_LEN, interv_answer_len=INTERV_ANSWER_LEN):
    """Read one hop level's QA file, keep rows whose answer count is in window, subsample."""
    qa_path = os.path.join(DATA_DIR, f"{hops}-hop", "vanilla", f"{SPLIT_FILES[split]}.txt")
    qtype_path = os.path.join(DATA_DIR, f"{hops}-hop", f"{SPLIT_FILES[split]}_qtype.txt")
    with open(qa_path) as f:
        lines = f.read().splitlines()
    with open(qtype_path) as f:
        qtypes = f.read().splitlines()

    lo = max(1, answer_len - interv_answer_len)
    hi = answer_len + interv_answer_len

    records = []
    for line, qtype in zip(lines, qtypes):
        question, _, answers = line.partition("\t")
        answers = answers.split("|")
        if not lo <= len(answers) <= hi:
            continue
        topic = TOPIC_RE.search(question)
        if topic is None:
            continue
        answers = sorted(answers)
        records.append({
            # Brackets are stripped so `retrieved` is not handed the topic entity for
            # free; `gold` uses the annotated span, which the prompt never shows.
            "question": TOPIC_RE.sub(r"\1", question),
            "topic": topic.group(1),
            "answers": answers,
            "answer_text": ANSWER_SEP.join(answers),
            "chain": qtype_chain(qtype),
            "nsteps": hops,
            "task": derive_task(hops),
        })
    return rng.sample(records, min(n_samples, len(records)))


def build_rows(records, kb, rng, desc, strategies=STRATEGIES):
    rows, dropped, covered = [], 0, {strategy: [0, 0] for strategy in strategies}
    for record in tqdm(records, desc=desc):
        for strategy in strategies:
            built = kb.build_subgraph(record, strategy, max_triples=MAX_TRIPLES,
                                      max_nodes=MAX_NODES, rng=rng)
            if built is None:
                dropped += 1
                continue
            text, nnodes, nedges = built
            complete = all(answer in text for answer in record["answers"])
            covered[strategy][1] += 1
            covered[strategy][0] += complete
            # An unanswerable gold example would only teach the model to guess;
            # under `retrieved`, missing evidence is the phenomenon being measured.
            if strategy == "gold" and not complete:
                dropped += 1
                continue
            rows.append({
                "algorithm": strategy,
                "task": record["task"],
                "nnodes": str(nnodes),
                "nedges": str(nedges),
                "nsteps": str(record["nsteps"]),
                "nanswers": str(len(record["answers"])),
                "question": f"{text}Q: {record['question']}\nA: ",
                "answer": f"{record['answer_text']}.",
            })
    rates = " | ".join(f"{s} answers-in-graph {100 * hit / max(1, total):.1f}%"
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
            f"Question:\n{record['question']}\nA: {record['answer_text']}."
        )
    return bank


def attach_examples(rows, bank, rng):
    for row in rows:
        pool = bank[row["task"]]
        row["example"] = [rng.choice(pool) for _ in range(N_EXAMPLES_PER_SAMPLE)]


def main(strategies=STRATEGIES, answer_len=ANSWER_LEN, interv_answer_len=INTERV_ANSWER_LEN,
         output_dir=OUTPUT_DIR, seed=SEED):
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown seeding strategies: {unknown}. "
                         f"Expected a subset of {STRATEGIES}.")

    rng = random.Random(seed)
    print(f"Loading KB from {KB_PATH} ...")
    kb = KBIndex(KB_PATH)

    sizes = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}
    split_records, split_rows = {}, {}
    for split, total in sizes.items():
        records = []
        for hops in HOPS:
            records.extend(load_records(hops, split, total // len(HOPS), rng,
                                        answer_len, interv_answer_len))
        rng.shuffle(records)
        split_records[split] = records
        split_rows[split] = build_rows(records, kb, rng, desc=f"subgraphs [{split}]",
                                       strategies=strategies)

    bank = build_example_bank(split_records["train"])
    for model_type in MODEL_TYPES:
        for split, rows in split_rows.items():
            rows = [dict(row) for row in rows]
            if model_type == "meta-icl":
                attach_examples(rows, bank, rng)
            data = [{"id": f"{split}_{i}", **row} for i, row in enumerate(rows)]
            output_file = os.path.join(output_dir, model_type, "standard", f"{split}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"{model_type}/{split}: {len(data)} rows -> {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES,
                        help="seeding strategies to emit; each becomes an `algorithm` value")
    parser.add_argument("--answer-len", type=int, default=ANSWER_LEN,
                        help="target number of answers per question")
    parser.add_argument("--interv", type=int, default=INTERV_ANSWER_LEN,
                        help="half-width of the accepted window around --answer-len")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.strategies, args.answer_len, args.interv, args.output_dir, args.seed)
