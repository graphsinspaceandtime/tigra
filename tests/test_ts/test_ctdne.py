import sys
import os

workers = int(os.environ.get("NUM_CPUS", 4))

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.abspath("."))

sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph, Graph, DynamicAggregateGraph

from tigra.embeddings.embedding import FSEmbedding
from tigra.embeddings.sota.ctdne_wrapper import Ctdne


from tigra.core.greval import GraphReconstructionEval
import pickle

from tigra.evaluation.classification_eval import eval
from tigra.evaluation.classification.neural_network_classifier import (
    NeuralNetworkClassifier,
)
from tigra.evaluation.clbase.skl import skl_classifiers_factory

# from tigra.evaluation import lid_eval


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

    es = dyn.get_graph_sequence()

    dag = DynamicAggregateGraph("radoslaw", dyn, directed=False)
    print("DIRECTED = ", dag.get_aggregate_graph().is_directed())

    ctdne = Ctdne(10, workers=workers)
    ctdne.make_embedding(dyn)
    folder_name = "plane_emb"
    ctdne.save_embedding(folder_name)

    org_graphs = dyn.get_graph_sequence()
    for i in range(len(org_graphs)):
        print("SNAPSHOT", i)
        emb = ctdne.embeddings[i]
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

    # print("TESTING TUNER")
    # start_time = time.time()
    # tuner = Dynnode2vec_Tuner(
    #     last_graph,
    #     dyn,
    #     [0.5, 1.0, 2.0, 4.0],
    #     [0.5, 1.0, 2.0, 4.0],
    #     num_runs=10,
    #     parallel_processes=16,
    #     walk_length=64,
    #     n_walks_per_node=10,
    # )
    # # tuner = Dynnode2vec_Tuner(last_graph, dg, [0.5, 1.0], [0.5, 1.0], num_runs=5)
    # tuner.tune()
    # tuner.print_report()
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("TIME [sec]", elapsed_time)
