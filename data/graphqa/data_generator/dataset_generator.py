import sys, os
# os.chdir("../../..")
print("dir", os.getcwd())

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
        for sample in samples:
            rows.append({
                "algorithm": algo,
                "task": task.__class__.__name__,
                "question": sample["question"],
                "answer": sample["answer"]
            })

print("rows")





# for split in SPLITS:
     

# with open(f"{OUTPUT_DIR}graphqa_dataset.json", "w") as f:
#     pass




