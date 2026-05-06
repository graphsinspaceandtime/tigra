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
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    "ia-primary-school-proximity-attr",
    # "ia-reality-call",s
    "radoslaw-email",
]


def anomalieseval(dataset, dg, emb_name, emb_path):
    emb = FSEmbedding(f"{emb_path}/{dataset}")

    os.makedirs(f"anomalies_eval/{emb_name}", exist_ok=True)

    os.makedirs(f"anomalies_eval/{emb_name}/plots", exist_ok=True)
    with open(f"anomalies_eval/{emb_name}/{dataset}.txt", "w", encoding="utf-8") as f:
        f.write(
            "SNAPSHOT_NUM,corr_top10_lof_cen,corr_top10_iso_cen,corr_top10_lof_btw,corr_top10_iso_btw,corr_top10_lof_egn,corr_top10_iso_egn\n"
        )
    anomaliesEval = AnomaliesEval(emb, dg, dataset, f"anomalies_eval/{emb_name}/{dataset}.txt")
    anomaliesEval.eval()


def anomalieseval_baseline(dataset, emb_name, emb_path):
    for name, factory in factories.items():
        print(name)

        res = factory.get_embedding(dataset)
        if res == None:
            return

        dn2v, dg = res
        emb = FSEmbedding(f"{emb_path}/{dataset}")

        os.makedirs(f"anomalies_eval/{emb_name}", exist_ok=True)
        with open(
            f"anomalies_eval/{emb_name}/{dataset}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(
                "SNAPSHOT_NUM,corr_top10_lof_cen,corr_top10_iso_cen,corr_top10_lof_btw,corr_top10_iso_btw,corr_top10_lof_egn,corr_top10_iso_egn\n"
            )
        anomaliesEval = AnomaliesEval(
            emb, dg, dataset, f"anomalies_eval/{emb_name}/{dataset}.txt"
        )
        anomaliesEval.eval()
        anomaliesEval.eval_peripheral_validation()


def load_novel_emb(dataset):
    es, resol = load_event_sequence(dataset)
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()
    return dg


if __name__ == "__main__":
    emb_algorithms = {
        "deep_lid": "emb/lid/deep_lid",
        "deep_hub": "emb/hub/deep_hub",
        "dn2v": "emb/dn2v",
    }

    os.makedirs("anomalies_eval", exist_ok=True)

    for emb_name, emb_path in emb_algorithms.items():
        print("emb", emb_name)
        for dataset in names:

            if "synt" not in dataset:
                print("DATASET", dataset)
                if emb_name == "dn2v":
                    anomalieseval_baseline(dataset, emb_name, emb_path)
                else:
                    dg = load_novel_emb(dataset)
                    anomalieseval(dataset, dg, emb_name, emb_path)
                print(
                    "......................................................................"
                )
