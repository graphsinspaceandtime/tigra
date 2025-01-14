"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to perform similarity search evaluation
"""

import sys

sys.path.append("..")

from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.evaluation.clseval import ClusteringEval

from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

dn2v_fact = get_dn2v_factory("/mnt/c/Users/babic/sources/tigra/dn2v.conf")


def lideval(dataset):
    global dn2v_fact
    res = dn2v_fact.get_embedding(dataset)
    if res == None:
        return

    dn2v, dg = res

    lide = ClusteringEval(dn2v, dg)
    lide.eval()


if __name__ == "__main__":

    for dataset in datasets:
        if "synt" not in dataset:
            print("DATASET", dataset)
            lideval(dataset)
            print(
                "......................................................................"
            )
