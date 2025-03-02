"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to perform clustering
"""

import sys

sys.path.append("..")

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
methods = ["dn2v_", "stwalk_", "ctdne_"]

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
        emb = FSEmbedding(f"emb/{name}_{dataset}")

        graph_sequence = dg.get_graph_sequence()

        for i in range(len(graph_sequence)):
            print("SNAPSHOT", i)
            snap = emb.embeddings[i]
            graph = graph_sequence[i]

            gre = GraphReconstructionEval(graph, snap)

            print("Precision", gre.get_avg_precision())
            print("Recall", gre.get_avg_recall())
            print("F1", gre.getF1())

        path = os.path.join("clustering_eval_3", f"{name}_{dataset}.txt")
        with open(path, "w", encoding="utf-8") as f:
            # f.write(
            #     "SNAPSHOT_NUM,GRAPH_ALG,EMBEDDING_ALG,GRAPH_NUM_COMMUNITIES,EMBEDDING_NUM_CLUSTERS,GRAPH_MODULARITY,GRAPH_SILHOUETTE,EMBEDDING_SILHOUETTE,NMI_GRAPH_EMBEDDING\n"
            # )
            f.write(
                "SNAPSHOT_NUM,GRAPH_ALG,GRAPH_MODULARITY, GRAPH_NUM_COMMUNITIES,EMBEDDING_ALG,EMBEDDING_NUM_CLUSTERS,EMBEDDING_SILHOUETTE,EMBEDDING_MOD\n"
            )
        lide = ClusteringEval(emb, dg, dataset, path)
        lide.eval()


if __name__ == "__main__":
    os.mkdir("clustering_eval_3")
    for dataset in datasets:
        # if "synt" not in dataset:
        if "radoslaw" in dataset:
            print("DATASET", dataset)
            clustereval(dataset)
            print(
                "......................................................................"
            )
