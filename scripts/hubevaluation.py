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

databank = Databank()
datasets = databank.get_keys()

dn2v_fact = get_dn2v_factory("../DN2V_SVC.conf")
stw_fact = get_stwalk_factory("../stwalk.conf")
ctdne_fact = get_ctdne_factory("../ctdne.conf")


def hubeval(dataset):
    global dn2v_fact
    res = dn2v_fact.get_embedding(dataset)
    if res == None:
        return

    dn2v, dg = res

    hube = HUBEval(dn2v, dg)
    hube.eval()


def hubeval_stwalk(dataset):
    global stw_fact
    res = stw_fact.get_embedding(dataset)
    if res == None:
        return

    stw, dg = res

    hube = HUBEval(stw, dg)
    hube.eval()


def hubeval_ctdne(dataset):
    global ctdne_fact
    res = ctdne_fact.get_embedding(dataset)
    if res == None:
        return

    ctdne, agg = res

    hube = HUBEval(ctdne, agg)
    hube.eval()


if __name__ == "__main__":
    for dataset in datasets:
        print("DATASET", dataset)
        if dataset != "ia-contacts_hypertext2009":
            hubeval_stwalk(dataset)
        print("......................................................................")
    
