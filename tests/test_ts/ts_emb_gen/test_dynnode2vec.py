import sys
import os
import time

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.abspath("."))

sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph, Graph
from tigra.embeddings.rndw.rndwbase import DynN2V
from tigra.embeddings.embedding import FSEmbedding

from tigra.core.greval import GraphReconstructionEval
import pickle

from tigra.evaluation.classification_eval import eval
from tigra.evaluation.clbase.skl import skl_classifiers_factory


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

    dn2v = DynN2V(100, 0.5, 4)
    dn2v.make_embedding(dyn)
    folder_name = "Plane_10_5_1.0_graph_graph_emb_dn2v"
    dn2v.save_embedding(folder_name)

    org_graphs = dyn.get_graph_sequence()
    for i in range(len(org_graphs)):
        print("SNAPSHOT", i)
        emb = dn2v.embeddings[i]
        graph = org_graphs[i]

        gre = GraphReconstructionEval(graph, emb)

        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())

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

    for classifier in skl_classifiers_factory():
        (score_accuracy, score_precision, score_recall, score_f1) = eval(
            emb, classifier
        )
        print(
            f"{classifier.__class__.__name__}\t{score_accuracy}\t{score_precision}\t{score_recall}\t{score_f1}"
        )
