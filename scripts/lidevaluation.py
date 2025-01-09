"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to perform LID-based evaluation
"""

import sys

sys.path.append("..")

from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.evaluation.lideval import LIDEval

from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

dn2v_fact = get_dn2v_factory("../DN2V_SVC.conf")


def lideval(dataset):
    global dn2v_fact
    res = dn2v_fact.get_embedding(dataset)
    if res == None:
        return

    dn2v, dg = res

    lide = LIDEval(dn2v, dg)
    lide.eval()


if __name__ == "__main__":

    for dataset in datasets:
        print("DATASET", dataset)
        lideval(dataset)
        print("......................................................................")
