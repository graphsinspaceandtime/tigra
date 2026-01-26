import sys
import os

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from tigra.embeddings.sota.factory import get_deep_lid_factory
from tigra.embeddings.embedding import FSEmbedding

from data.bank import Databank
from tigra.core.loaders import load_event_sequence
from tigra.core.greval import GraphReconstructionEval
from tigra.core.dgraph import DynamicGraph


deep_lid_fact = get_deep_lid_factory("../deep_lid.conf")

datasets = [
    "radoslaw-email",
    "fb-forum",
    "fb-messages",
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    "ia-primary-school-proximity-attr",
    "CollegeMsg",
    "email-Eu-core-temporal",
    "ia-reality-call",
]


for dataset in datasets:
    print("DATASET", dataset)
    deep_lid, dg = deep_lid_fact.get_embedding(dataset)
    deep_lid.save_embedding(dataset)

    e = FSEmbedding(dataset)

    databank = Databank()
    es, resol = load_event_sequence(dataset)
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    graph_sequence = dg.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb = e.embeddings[i]
        graph = graph_sequence[i]

        gre = GraphReconstructionEval(graph, emb)

        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())
