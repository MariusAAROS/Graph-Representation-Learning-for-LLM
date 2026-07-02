import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
from data.graphqa.data_generator import graph_generators, graph_tasks, graph_text_encoders
from data.graphqa.data_generator.graph_tasks import CycleCheck, \
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
from data.graphqa.data_generator.graph_text_encoders import adjacency_encoder, \
                                                        friendship_encoder, \
                                                        coauthorship_encoder, \
                                                        incident_encoder, \
                                                        social_network_encoder, \
                                                        expert_encoder
from sklearn.model_selection import train_test_split


N_GRAPHS = 5
DIRECTED = False
ALGORITHMS = ["er", "ba", "sbm", "sfn", "complete", "star", "path"]
TASKS = [CycleCheck(), EdgeExistence(), NodeCount(), 
         NodeDegree(), EdgeCount(), ConnectedNodes(), 
         DisconnectedNodes(), Reachability(), ShortestPath(), 
         TriangleCounting(), MaximumFlow()]
SEED = 42
ER_MIN_SPARSITY = 0.2
ER_MAX_SPARSITY = 0.8
OUTPUT_DIR = "data/graphqa/"
SPLITS = ["train", "val", "test"]
AVAILABLE_ENCODING_METHODS = ["adjacency", "incident", "friendship", 
                    "south_park", "got", "politician",
                    "social_network", "expert", "coauthorship",
                    "random", "nx_node_name"]
ENCODING_METHODS = AVAILABLE_ENCODING_METHODS[0]
TRAIN_SPLIT = 0.7
TEST_SPLIT = (1 - TRAIN_SPLIT) / 2
VAL_SPLIT = TEST_SPLIT

# samples = {}
rows = []
for algo in ALGORITHMS:
    graphs = graph_generators.generate_graphs(
        number_of_graphs=N_GRAPHS,
        algorithm=algo,
        directed=DIRECTED,
        random_seed=SEED,
        er_min_sparsity=ER_MIN_SPARSITY,
        er_max_sparsity=ER_MAX_SPARSITY
    )

    # samples[algo] = {}
    for task in TASKS:
        samples = task.prepare_examples_dict(
            graphs=graphs,
            generator_algorithms=[algo] * len(graphs),
            encoding_method=ENCODING_METHODS
        )
        for sample in samples.values():
            rows.append({
                # "id": sample["id"],
                "algorithm": algo,
                "task": task.__class__.__name__,
                "nnodes": sample["nnodes"],
                "nedges": sample["nedges"],
                "question": sample["question"],
                "answer": sample["answer"]
            })
x_keys = [key for key in rows[0].keys() if key != "answer"]
X = [{key: row[key] for key in x_keys} for row in rows]
y = [row["answer"] for row in rows]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=TRAIN_SPLIT, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

train_data = [{"id": f"train_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_train, y_train))]
val_data = [{"id": f"val_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_val, y_val))]
test_data = [{"id": f"test_{i}", **x, "answer": y} for i, (x, y) in enumerate(zip(X_test, y_test))]

for split, data in zip(SPLITS, [train_data, val_data, test_data]):
    output_file = os.path.join(OUTPUT_DIR, f"{split}.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)