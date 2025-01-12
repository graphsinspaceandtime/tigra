"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to perform LID-based evaluation
"""

import sys

sys.path.append("..")

from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.embeddings.sota.factory import get_stwalk_factory
from tigra.embeddings.sota.factory import get_ctdne_factory
from tigra.evaluation.lideval import LIDEval

from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

dn2v_fact = get_dn2v_factory("../DN2V_SVC.conf")
stw_fact = get_stwalk_factory("../stwalk.conf")
ctdne_fact = get_ctdne_factory("../ctdne.conf")


def lideval_dn2v(dataset):
    global dn2v_fact
    res = dn2v_fact.get_embedding(dataset)
    if res == None:
        return

    dn2v, dg = res

    lide = LIDEval(dn2v, dg)
    lide.eval()

def lideval_stwalk(dataset):
    global stw_fact
    res = stw_fact.get_embedding(dataset)
    if res == None:
        return

    stw, dg = res

    lide = LIDEval(stw, dg)
    lide.eval()

def lideval_ctdne(dataset):
    global ctdne_fact
    res = ctdne_fact.get_embedding(dataset)
    if res == None:
        return

    ctdne, agg = res

    lide = LIDEval(ctdne, agg)
    lide.eval()

if __name__ == "__main__":

    for dataset in datasets:
        print("DATASET", dataset)
        if dataset != "ia-contacts_hypertext2009":
            lideval_ctdne(dataset)
        print("......................................................................")
