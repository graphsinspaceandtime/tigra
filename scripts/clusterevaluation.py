"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to perform clustering
"""

import sys

sys.path.append("..")

from tigra.core.dgraph import DynamicAggregateGraph, DynamicGraph
from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.embeddings.sota.factory import get_stwalk_factory
from tigra.embeddings.sota.factory import get_ctdne_factory
from tigra.embeddings.embedding import FSEmbedding
from tigra.evaluation.clustering_eval import ClusteringEval
from tigra.core.greval import GraphReconstructionEval

from data.bank import Databank
import os

databank = Databank()
datasets = databank.get_keys()

factories = {
    "dn2v": get_dn2v_factory("DN2V_SVC.conf"),
    # "stwalk": get_stwalk_factory("stwalk.conf"),
    # "ctdne": get_ctdne_factory("ctdne.conf"),
}


def clustereval(dataset):
    for name, factory in factories.items():
        print(name)
        res = factory.get_embedding(dataset)
        if res == None:
            return
        dn2v, dg = res
        emb = FSEmbedding(f"emb2/{name}_{dataset}")

        if isinstance(dg, DynamicAggregateGraph):  # ctdne
            graph = dg.get_aggregate_graph()
            embd = emb.embeddings[0]
            gre = GraphReconstructionEval(graph, embd)
            print("Precision", gre.get_avg_precision())
            print("Recall", gre.get_avg_recall())
            print("F1", gre.getF1())

            path = os.path.join("clustering_eval_ctdne", f"{name}_{dataset}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("METHOD,LP,,Louvain,,GMO,,KMEANS,,,DBSCAN,,,AGG,,\n")
                f.write("SNAPSHOT_NUM,Q,NC,Q,NC,Q,NC,Q,NC,SIL,Q,NC,SIL,Q,NC,SIL\n")
            lide = ClusteringEval(emb, dg, dataset, path)
            lide.eval()

        else:  # dn2vec
            graph_sequence = dg.get_graph_sequence()

            print(f"graph sequence num: {len(graph_sequence)}\n")
            print(f"emb num: {len(emb.embeddings)}\n")

            for i in range(len(graph_sequence)):
                print("SNAPSHOT", i)

                snap = emb.embeddings[i]
                graph = graph_sequence[i]
                gre = GraphReconstructionEval(graph, snap)
                print("Precision", gre.get_avg_precision())
                print("Recall", gre.get_avg_recall())
                print("F1", gre.getF1())
                print(f"Number of nodes:{gre.graph.number_of_nodes()}")
            
            path = os.path.join("clustering_eval_dn2vec_n", f"{name}_{dataset}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("METHOD,LP,,Louvain,,GMO,,KMEANS,,,AGG,,\n")
                f.write("SNAPSHOT_NUM,Q,NC,Q,NC,Q,NC,Q,NC,SIL,Q,NC,SIL\n")
            lide = ClusteringEval(emb, dg, dataset, path)
            lide.eval()


if __name__ == "__main__":
    os.mkdir("clustering_eval_dn2vec_n")
    os.mkdir("clustering_eval_dn2vec_nmi")

    for dataset in datasets:
        if "synt" not in dataset:
        # if "radoslaw" in dataset:
            print("DATASET", dataset)
            clustereval(dataset)
            print(
                "......................................................................"
            )
