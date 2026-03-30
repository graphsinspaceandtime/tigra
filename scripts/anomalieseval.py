"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to perform clustering
"""

import sys
import os


sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph

from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.embeddings.sota.factory import get_stwalk_factory
from tigra.embeddings.sota.factory import get_ctdne_factory
from tigra.embeddings.embedding import FSEmbedding
from tigra.evaluation.anmeval import AnomaliesEval
from tigra.core.loaders import load_event_sequence

from data.bank import Databank
import os

databank = Databank()
datasets = databank.get_keys()

# methods = ["dn2v_", "stwalk_", "ctdne_"]

factories = {
    "dn2v": get_dn2v_factory("configs/DN2V_SVC.conf"),
    # "stwalk": get_stwalk_factory("stwalk.conf"),
    # "ctdne": get_ctdne_factory("ctdne.conf"),
}

names = [
    "CollegeMsg",
    "email-Eu-core-temporal",
    "fb-forum",
    "fb-messages",
    "ia_contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    "ia-primary-school-proximity-attr",
    "ia-reality-call",
    "radoslaw-email",
]


def anomalieseval(dataset, dg):
    emb = FSEmbedding(f"emb/lid/deep_lid/{dataset}")

    path = os.path.join("anomalies_eval_top10", f"lof_{dataset}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "SNAPSHOT_NUM,corr_top10_lof_cen,corr_top10_iso_cen,corr_top10_lof_btw,corr_top10_iso_btw,corr_top10_lof_egn,corr_top10_iso_egn\n"
        )
    lide = AnomaliesEval(emb, dg, dataset, path)
    lide.eval()


def anomalieseval_baseline(dataset):
    for name, factory in factories.items():
        print(name)

        res = factory.get_embedding(dataset)
        if res == None:
            return

        dn2v, dg = res
        emb = FSEmbedding(f"emb/{name}_{dataset}")

        path = os.path.join("anomalies_eval_top10_dyn2v", f"{name}_{dataset}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "SNAPSHOT_NUM,corr_top10_lof_cen,corr_top10_iso_cen,corr_top10_lof_btw,corr_top10_iso_btw,corr_top10_lof_egn,corr_top10_iso_egn\n"
            )
        lide = AnomaliesEval(emb, dg, dataset, path)
        lide.eval()


def load_novel_emb(dataset):
    es, resol = load_event_sequence(dataset)
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()
    return dg


if __name__ == "__main__":
    if not os.path.exists("anomalies_eval_top10_dyn2v"):
        os.mkdir("anomalies_eval_top10_dyn2v")
    for dataset in datasets:

        if "synt" not in dataset:
            print("DATASET", dataset)
            # dg = load_novel_emb(dataset)
            # anomalieseval(dataset, dg)
            anomalieseval_baseline(dataset)
            print(
                "......................................................................"
            )
