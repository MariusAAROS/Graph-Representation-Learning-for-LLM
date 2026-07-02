import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
from datasets.graphqa import graph_generators
from datasets.graphqa.graph_tasks import CycleCheck, \
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
from datasets.graphqa.graph_text_encoders import adjacency_encoder, \
                                                        friendship_encoder, \
                                                        coauthorship_encoder, \
                                                        incident_encoder, \
                                                        social_network_encoder, \
                                                        expert_encoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

N_GRAPHS = 50
DIRECTED = False
ALGORITHMS = ["er", "ba", "sbm", "sfn", "complete", "star", "path"]
TASKS = [CycleCheck(), EdgeExistence(), NodeCount(), 
         NodeDegree(), EdgeCount(), ConnectedNodes(), 
         DisconnectedNodes(), Reachability(), ShortestPath(), 
         TriangleCounting(), MaximumFlow()]
STR_TASKS = [task.__class__.__name__ for task in TASKS]
N_EXAMPLES_PER_GRAPH = 32
N_EXAMPLES = N_GRAPHS * len(TASKS) * N_EXAMPLES_PER_GRAPH
SEED = 42
ER_MIN_SPARSITY = 0.2
ER_MAX_SPARSITY = 0.8
MODEL_TYPE = "meta-icl" # meta-icl, baseline
OUTPUT_DIR = "data/graphqa/"

SPLITS = ["train", "val", "test"]
AVAILABLE_ENCODING_METHODS = ["adjacency", "incident", "friendship", 
                    "south_park", "got", "politician",
                    "social_network", "expert", "coauthorship",
                    "random", "nx_node_name"]
ENCODING_METHODS = AVAILABLE_ENCODING_METHODS[0]
TRAIN_SPLIT = 0.8
TEST_SPLIT = (1 - TRAIN_SPLIT) / 2
VAL_SPLIT = TEST_SPLIT
OOD_TEST = True
OOD_SELECTION = "lastn" # random | lastn | cherrypick

# CycleCheck, EdgeExistence, NodeCount, 
# NodeDegree, EdgeCount, ConnectedNodes, DisconnectedNodes, 
# Reachability, ShortestPath, TriangleCounting, MaximumFlow
CP_TRAIN_TASKS = [...]
CP_TEST_TASKS = [t for t in STR_TASKS if t not in CP_TRAIN_TASKS]

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
                    "answer": sample["answer"]
                })
else:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'meta-icl' or 'baseline'.")

x_keys = [key for key in rows[0].keys() if key != "answer"]
X = [{key: row[key] for key in x_keys} for row in rows]
y = [row["answer"] for row in rows]

if OOD_TEST:
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
    

    test_permutations = np.random.permutation(len(test_indices))
    test_indices = [test_indices[i] for i in test_permutations]
    X_train = np.array(X_train)[train_permutations]
    y_train = np.array(y_train)[train_permutations]

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
else:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=TRAIN_SPLIT, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

train_data = [{"id": f"train_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_train, y_train))]
val_data = [{"id": f"val_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_val, y_val))]
test_data = [{"id": f"test_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_test, y_test))]

print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")
for split, data in zip(SPLITS, [train_data, val_data, test_data]):
    output_file = os.path.join(OUTPUT_DIR, MODEL_TYPE, 
                               "ood" if OOD_TEST else "standard",
                               f"{split}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
print("Datasets saved successfully.")