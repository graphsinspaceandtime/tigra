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
