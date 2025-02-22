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
from tigra.evaluation.anmeval import AnomaliesEval

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


def anomalieseval(dataset):
    for name, factory in factories.items():
        print(name)

        res = factory.get_embedding(dataset)
        if res == None:
            return

        dn2v, dg = res
        emb = FSEmbedding(f"emb/{name}_{dataset}")

        path = os.path.join("anomalies_eval", f"lof_{name}_{dataset}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "SNAPSHOT_NUM,corr_lof_cen,p_lof_cen,corr_iso_cen,p_iso_cen,corr_aen_cen,p_aen_cen,corr_lof_btw,p_lof_btw,corr_iso_btw,p_iso_btw,corr_aen_btw,p_aen_btw,corr_lof_egn,p_lof_egn,corr_iso_egn,p_iso_egn,corr_aen_egn,p_aen_egn,\n"
            )
        lide = AnomaliesEval(emb, dg, dataset, path)
        lide.eval()


if __name__ == "__main__":
    os.mkdir("anomalies_eval")
    for dataset in datasets:
        if "synt" not in dataset:
            print("DATASET", dataset)
            anomalieseval(dataset)
            print(
                "......................................................................"
            )
