import sys, os
# Repo root, so that `src.` resolves (a bare `datasets.` would hit HF datasets).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
import random
from src.datasets.graphqa import graph_generators
from src.datasets.graphqa.graph_tasks import CycleCheck, \
                                                    EdgeExistence, \
                                                    NodeCount, \
                                                    NodeDegree, \
                                                    EdgeCount, \
                                                    ConnectedNodes, \
                                                    DisconnectedNodes, \
                                                    Reachability, \
                                                    ShortestPath, \
                                                    TriangleCounting, \
                                                    MaximumFlow
from src.datasets.graphqa.graph_text_encoders import adjacency_encoder, \
                                                        friendship_encoder, \
                                                        coauthorship_encoder, \
                                                        incident_encoder, \
                                                        social_network_encoder, \
                                                        expert_encoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import argparse

SEED = 42
ER_MIN_SPARSITY = 0.2
ER_MAX_SPARSITY = 0.8
DIRECTED = False
ALGORITHMS = ["er", "ba", "sbm", "sfn", "complete", "star", "path"]
TASKS = [CycleCheck(), EdgeExistence(), NodeCount(), 
         NodeDegree(), EdgeCount(), ConnectedNodes(), 
         DisconnectedNodes(), Reachability(), ShortestPath(), 
         TriangleCounting(), MaximumFlow()]
STR_TASKS = [task.__class__.__name__ for task in TASKS]
SPLITS = ["train", "val", "test"]
AVAILABLE_ENCODING_METHODS = ["adjacency", "adjacency_wl", "adjacency_wl_shuf",
                    "incident", "friendship",
                    "south_park", "got", "politician",
                    "social_network", "expert", "coauthorship",
                    "random", "nx_node_name"]
# CycleCheck, EdgeExistence, NodeCount, 
# NodeDegree, EdgeCount, ConnectedNodes, DisconnectedNodes, 
# Reachability, ShortestPath, TriangleCounting, MaximumFlow
CP_TRAIN_TASKS = [...]
CP_TEST_TASKS = [t for t in STR_TASKS if t not in CP_TRAIN_TASKS]

parser = argparse.ArgumentParser(description="Generate GraphQA splits.")
parser.add_argument("--model-type", default="meta-icl", choices=["meta-icl", "baseline"])
# pool: all tasks, split by sample count, held-out task chosen at runtime (LOTO)
#       -> {output-dir}/{dataset-config}/ood_pool/
# ood:  static held-out tasks (see --ood-selection) -> .../ood/
# standard: all tasks, random split -> .../standard/
# train_only: all tasks, no split, everything -> .../train_only/train.json
#             (large training pools; pair with --exclude-json to keep existing val/test clean)
parser.add_argument("--split-mode", default="pool", choices=["pool", "ood", "standard", "train_only"])
parser.add_argument("--ood-selection", default="lastn", choices=["random", "lastn", "cherrypick"])
parser.add_argument("--n-graphs", type=int, default=50, help="graphs per generator algorithm")
parser.add_argument("--n-examples-per-graph", type=int, default=32)
parser.add_argument("--encoding", default="adjacency", choices=AVAILABLE_ENCODING_METHODS)
parser.add_argument("--train-split", type=float, default=0.8,
                    help="train fraction; the rest is halved into val and test")
parser.add_argument("--stratify", action="store_true",
                    help="stratify pool/standard splits by task (exact per-task counts)")
parser.add_argument("--output-dir", default="data/graphqa/")
parser.add_argument("--dataset-config", default=None,
                    help="output subfolder, i.e. cfg.dataset.dataset_config (default: model type)")
parser.add_argument("--seed", type=int, default=SEED,
                    help="seed for graph generation and splits (default reproduces the original data)")
parser.add_argument("--dedup", action="store_true",
                    help="drop exact (question, answer) repeats, keeping the first occurrence")
parser.add_argument("--exclude-json", nargs="+", default=[],
                    help="drop rows whose (question, answer) appears in any of these JSON files")
args = parser.parse_args()
SEED = args.seed

N_GRAPHS = args.n_graphs
N_EXAMPLES_PER_GRAPH = args.n_examples_per_graph
N_EXAMPLES = N_GRAPHS * len(TASKS) * N_EXAMPLES_PER_GRAPH
MODEL_TYPE = args.model_type
OUTPUT_DIR = args.output_dir
DATASET_CONFIG = args.dataset_config or MODEL_TYPE
ENCODING_METHODS = args.encoding
TRAIN_SPLIT = args.train_split
TEST_SPLIT = (1 - TRAIN_SPLIT) / 2
OOD_TEST = args.split_mode == "ood"
OOD_SELECTION = args.ood_selection
OOD_POOL_MODE = args.split_mode == "pool"
TRAIN_ONLY = args.split_mode == "train_only"


def _edges(sample):
    return [[int(u), int(v)] for u, v in sample["graph"].edges()]


if MODEL_TYPE == "meta-icl":
    print(f"Generating {N_GRAPHS*len(ALGORITHMS)*len(TASKS)} samples for {len(ALGORITHMS)} algorithms and {len(TASKS)} tasks.\n\
          Generating {N_EXAMPLES_PER_GRAPH} examples per graph for a total of {N_EXAMPLES} examples.")
    rows = []
    for algo in tqdm(ALGORITHMS, desc="Algorithms"):
        graphs = graph_generators.generate_graphs(
            number_of_graphs=N_GRAPHS,
            algorithm=algo,
            directed=DIRECTED,
            random_seed=SEED,
            er_min_sparsity=ER_MIN_SPARSITY,
            er_max_sparsity=ER_MAX_SPARSITY
        )

        example_graphs = [
            graph_generators.generate_graphs(
                number_of_graphs=N_EXAMPLES_PER_GRAPH,
                algorithm=algo,
                directed=DIRECTED,
                random_seed=SEED,
                er_min_sparsity=ER_MIN_SPARSITY,
                er_max_sparsity=ER_MAX_SPARSITY
            )
            for _ in range(N_GRAPHS)
        ]

        for task in tqdm(TASKS, desc=f"Tasks for {algo}"):
            samples = task.prepare_examples_dict(
                graphs=graphs,
                generator_algorithms=[algo] * len(graphs),
                encoding_method=ENCODING_METHODS
            )
            example_samples = [
                task.prepare_examples_dict(
                    graphs=example_graphs[i],
                    generator_algorithms=[algo] * len(example_graphs[i]),
                    encoding_method=ENCODING_METHODS
                )
                for i in range(N_GRAPHS)
            ]
            formated_examples = [f"Question:\n{ex['question']}" + ex['answer']
                                 for fex in example_samples for ex in fex.values()]
            formated_examples = []
            for fexamples in example_samples:
                temp = []
                for ex in fexamples.values():
                    temp.append(f"Question:\n{ex['question']}" + ex['answer'])
                formated_examples.append(temp)

            for sample, current_examples in zip(samples.values(), formated_examples):
                rows.append({
                    "algorithm": algo,
                    "task": task.__class__.__name__,
                    "nnodes": sample["nnodes"],
                    "nedges": sample["nedges"],
                    "question": sample["question"],
                    "edges": _edges(sample),
                    "example": current_examples,
                    "answer": sample["answer"]
                })
elif MODEL_TYPE == "baseline": 
    print(f"Generating {N_GRAPHS*len(ALGORITHMS)*len(TASKS)} samples for {len(ALGORITHMS)} algorithms and {len(TASKS)} tasks.")
    rows = []
    for algo in tqdm(ALGORITHMS, desc="Algorithms"):
        graphs = graph_generators.generate_graphs(
            number_of_graphs=N_GRAPHS,
            algorithm=algo,
            directed=DIRECTED,
            random_seed=SEED,
            er_min_sparsity=ER_MIN_SPARSITY,
            er_max_sparsity=ER_MAX_SPARSITY
        )

        # samples[algo] = {}
        for task in tqdm(TASKS, desc=f"Tasks for {algo}"):
            samples = task.prepare_examples_dict(
                graphs=graphs,
                generator_algorithms=[algo] * len(graphs),
                encoding_method=ENCODING_METHODS
            )
            for sample in samples.values():
                rows.append({
                    "algorithm": algo,
                    "task": task.__class__.__name__,
                    "nnodes": sample["nnodes"],
                    "nedges": sample["nedges"],
                    "question": sample["question"],
                    "edges": _edges(sample),
                    "answer": sample["answer"]
                })
else:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'meta-icl' or 'baseline'.")

if args.dedup or args.exclude_json:
    excluded = set()
    for path in args.exclude_json:
        with open(path) as f:
            excluded.update((r["question"], r["answer"]) for r in json.load(f))
    seen, kept, n_excl, n_dup = set(), [], 0, 0
    for row in rows:
        key = (row["question"], row["answer"])
        if key in excluded:
            n_excl += 1
            continue
        if args.dedup:
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
        kept.append(row)
    print(f"Filtering: {len(rows)} -> {len(kept)} rows ({n_excl} excluded, {n_dup} duplicates)")
    rows = kept

from collections import Counter
print("Per algorithm:", dict(Counter(r["algorithm"] for r in rows)))
print("Per task:", dict(Counter(r["task"] for r in rows)))

x_keys = [key for key in rows[0].keys() if key != "answer"]
X = [{key: row[key] for key in x_keys} for row in rows]
y = [row["answer"] for row in rows]

def _split(X, y):
    strat = [x["task"] for x in X] if args.stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=TRAIN_SPLIT, random_state=SEED, stratify=strat)
    strat = [x["task"] for x in X_temp] if args.stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=strat)
    return X_train, X_val, X_test, y_train, y_val, y_test


if TRAIN_ONLY:
    # No split: the whole pool is training data. Shuffle so the file is not ordered by algorithm/task.
    perm = np.random.RandomState(SEED).permutation(len(X))
    X_train, y_train = [X[i] for i in perm], [y[i] for i in perm]
    X_val, X_test, y_val, y_test = [], [], [], []
    output_subdir = "train_only"
elif OOD_POOL_MODE:
    # Keep ALL tasks; split by sample count only. Which task is held out is
    # decided at runtime by the trainer (see runner.py + dataset.ood_task).
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)
    output_subdir = "ood_pool"
elif OOD_TEST:
    n_ood = int(len(TASKS) * TEST_SPLIT)
    if OOD_SELECTION == "random":
        random.seed(SEED)
        train_tasks = random.sample(TASKS, len(TASKS) - n_ood)
        test_tasks = [task for task in TASKS if task not in train_tasks]
    elif OOD_SELECTION == "lastn":
        train_tasks = TASKS[:-n_ood]
        test_tasks = TASKS[-n_ood:]
    elif OOD_SELECTION == "cherrypick":
        train_tasks = [task for task in TASKS if task.__class__.__name__ in CP_TRAIN_TASKS]
        test_tasks = [task for task in TASKS if task.__class__.__name__ in CP_TEST_TASKS]
    else:
        raise ValueError(f"Invalid OOD_SELECTION: {OOD_SELECTION}. Must be 'random', 'lastn', or 'cherrypick'.")
    train_tasks_names = [task.__class__.__name__ for task in train_tasks]
    test_tasks_names = [task.__class__.__name__ for task in test_tasks]
    train_indices = [i for i, row in enumerate(rows) if row["task"] in train_tasks_names]
    test_indices = [i for i, row in enumerate(rows) if row["task"] in test_tasks_names]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_temp = [X[i] for i in test_indices]
    y_temp = [y[i] for i in test_indices]
    
    train_permutations = np.random.permutation(len(train_indices))
    X_train = np.array(X_train)[train_permutations]
    y_train = np.array(y_train)[train_permutations]

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
    output_subdir = "ood"
else:
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)
    output_subdir = "standard"

train_data = [{"id": f"train_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_train, y_train))]
val_data = [{"id": f"val_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_val, y_val))]
test_data = [{"id": f"test_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_test, y_test))]

print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")
for split, data in zip(SPLITS, [train_data, val_data, test_data]):
    if TRAIN_ONLY and split != "train":
        continue
    output_file = os.path.join(OUTPUT_DIR, DATASET_CONFIG, 
                               output_subdir,
                               f"{split}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=None if TRAIN_ONLY else 4)
print("Datasets saved successfully.")