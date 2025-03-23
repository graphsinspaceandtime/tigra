"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to generate & save embeddings
"""

import sys

sys.path.append("..")

from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.embeddings.sota.factory import get_stwalk_factory
from tigra.embeddings.sota.factory import get_ctdne_factory

from tigra.embeddings.embedding import FSEmbedding
from tigra.core.greval import GraphReconstructionEval

from data.bank import Databank
import os

databank = Databank()
datasets = databank.get_keys()

factories = {
    "dn2v": get_dn2v_factory("DN2V_SVC.conf"),
    # "stwalk": get_stwalk_factory("../stwalk.conf"),
    # "ctdne": get_ctdne_factory("../ctdne.conf"),
}


def save_to_file(dataset):
    for name, factory in factories.items():
        res = factory.get_embedding(dataset)
        if res == None:
            return

        emb, dg = res
        emb.save_embedding(f"emb2/{name}_{dataset}")


def gre_eval(graph_sequence, emb):
    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)

        snap = emb.embeddings[i]
        graph = graph_sequence[i]
        gre = GraphReconstructionEval(graph, snap)
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())


def gen_dn2v_emb(dataset):
    dn2v_fact = get_dn2v_factory("../DN2V_SVC.conf")
    emb, dg = dn2v_fact.get_embedding(dataset)
    emb.save_embedding(f"emb/dn2v_{dataset}")
    gre_eval(dg.get_graph_sequence(), emb)
    return dg


if __name__ == "__main__":
    # dg = gen_dn2v_emb("ia-enron-employees")

    # emb = FSEmbedding(f"emb/dn2v_ia-enron-employees")
    # graph_sequence = dg.get_graph_sequence()
    # print("\n\n\nEVAL-FROM-LOAD")
    # gre_eval(graph_sequence, emb)

    os.mkdir("emb2")

    for dataset in datasets:
        if "synt" not in dataset:
            print("DATASET", dataset)
            save_to_file(dataset)
            print(
                "......................................................................"
            )
