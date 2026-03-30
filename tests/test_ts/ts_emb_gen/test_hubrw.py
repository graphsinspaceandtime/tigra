import sys
import os
import pickle

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph, Graph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.embedding import FSEmbedding
from tigra.embeddings.rndw.hubrw import DynDeepHubWalk

from tigra.core.greval import GraphReconstructionEval
import pickle


def load_tigra_dgraph(fname):
    with open(fname, "rb") as f:
        tigra_graphs = []
        graphs = pickle.load(f)
        for i, graph in enumerate(graphs):
            tigra_graphs.append(Graph(f"snapshot {i}", nx_graph=graph, labeled=True))
        dyn = DynamicGraph(list_of_graphs=tigra_graphs)
        dyn.info()

        return dyn


def process_dataset(pkl_path, log_file):
    prefix = os.path.basename(pkl_path).replace("_graph.pkl", "")
    folder_name = f"{prefix}_emb_hub"

    dyn = load_tigra_dgraph(pkl_path)
    dyn.info()

    deep_lid = DynDeepHubWalk(dim=25)
    deep_lid.make_embedding(dyn)
    deep_lid.save_embedding(folder_name)

    graph_sequence = dyn.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb = deep_lid.embeddings[i]
        graph = graph_sequence[i]

        gre = GraphReconstructionEval(graph, emb)
        precision = gre.get_avg_precision()
        recall = gre.get_avg_recall()
        f1 = gre.getF1()

        line = f"{prefix},{i},{precision:.4f},{recall:.4f},{f1:.4f}\n"
        print(line, end="")
        log_file.write(line)

        fn = str(i) + ".labels"
        path = os.path.join(folder_name, fn)
        graph.export_labels(path)


if __name__ == "__main__":
    pkl_files = [
        # "arrow_head_graph.pkl",
        # "beef_graph.pkl",
        # "beetle_fly_graph.pkl",
        # "herring_graph.pkl",
        "adiac_dynamic_graph.pkl"
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "hub_embedding_results.txt")

    with open(output_path, "w") as log_file:
        for pkl_path in pkl_files:
            process_dataset(pkl_path, log_file)
