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


if __name__ == "__main__":

    dyn = load_tigra_dgraph("Plane_10_5_1.0_graph.pkl")
    dyn.info()

    deep_lid = DynDeepHubWalk(dim=25)
    deep_lid.make_embedding(dyn)
    folder_name = "Plane_10_5_1.0_graph_emb_deep_hub"
    deep_lid.save_embedding(folder_name)

    graph_sequence = dyn.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb = deep_lid.embeddings[i]
        graph = graph_sequence[i]

        gre = GraphReconstructionEval(graph, emb)

        print("Precision", gre.get_avg_precision(), gre.get_avg_precision())
        print("Recall", gre.get_avg_recall(), gre.get_avg_recall())
        print("F1", gre.getF1(), gre.getF1())

        fn = str(i) + ".labels"
        path = os.path.join(folder_name, fn)
        graph.export_labels(path)

    emb = FSEmbedding(folder_name)
    print("#embeddings", len(emb.embeddings))

    for e in emb:
        nodes = e.get_node_ids()
        node_0 = nodes[0]
        emb_0 = e[node_0]
        label_0 = e.get_label(node_0)
        print(node_0, "label", label_0, "emb dimension", len(emb_0))
        print(emb_0)
        print()
