"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to perform HUB-based evaluation
"""

import sys

sys.path.append("..")

from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.embeddings.sota.factory import get_stwalk_factory
from tigra.embeddings.sota.factory import get_ctdne_factory
from tigra.evaluation.hubeval import HUBEval

from data.bank import Databank
import os

databank = Databank()
datasets = databank.get_keys()

factories = {
    "dn2v": get_dn2v_factory("../DN2V_SVC.conf"),
    "stwalk": get_stwalk_factory("../stwalk.conf"),
    "ctdne": get_ctdne_factory("../ctdne.conf"),
}


def save_to_file(dataset):
    for name, factory in factories.items():
        res = factory.get_embedding(dataset)
        if res == None:
            return

        emb, dg = res
        emb.save_embedding(f"emb/{name}_{dataset}")


if __name__ == "__main__":
    os.mkdir("emb")

    for dataset in datasets:
        if "synt" not in dataset:
            print("DATASET", dataset)
            save_to_file(dataset)
            print(
                "......................................................................"
            )
