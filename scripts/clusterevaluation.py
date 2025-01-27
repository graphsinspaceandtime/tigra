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
from tigra.evaluation.clseval import ClusteringEval

from data.bank import Databank
import os

databank = Databank()
datasets = databank.get_keys()
methods = ["dn2v_", "stwalk_", "ctdne_"]

factories = {
    "dn2v": get_dn2v_factory("DN2V_SVC.conf"),
    "stwalk": get_stwalk_factory("stwalk.conf"),
    "ctdne": get_ctdne_factory("ctdne.conf"),
}


def clustereval(dataset):
    for name, factory in factories.items():
        print(name)

        res = factory.get_embedding(dataset)
        if res == None:
            return

        dn2v, dg = res
        emb = FSEmbedding(f"emb/{name}_{dataset}")

        path = os.path.join("clustering_eval", f"{name}_{dataset}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "SNAPSHOT_NUM,NUM_COMMUNITIES,NUM_REC_COMMUNITIES,MODULARITY,SIL1_GT,SIL2_COMMS,RG_MODULARITY,NUM_REC_LABELS,NMI_GT_RG,NMI_GT_ALGLBL,NMI_COMM_LBL_ALGLBL,NMI_COMM_LBL_ALGCOMM\n"
            )
            lide = ClusteringEval(emb, dg, dataset, f)
            lide.eval()


if __name__ == "__main__":
    os.mkdir("clustering_eval")
    for dataset in datasets:
        if "synt" not in dataset:
            print("DATASET", dataset)
            clustereval(dataset)
            print(
                "......................................................................"
            )
