import sys
import os

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.rndw.lidrw import DynN2V_LIDDelta
from tigra.embeddings.rndw.lidrw import DynN2V_LIDRW
from tigra.embeddings.rndw.lidrw import DynN2V_LIDRWDelta
from tigra.embeddings.rndw.rndwbase import DynN2V
from tigra.core.loaders import load_event_sequence
from tigra.embeddings.sota.factory import get_dn2v_factory


db = Databank()
datasets = db.get_keys()

dn2v_fact = get_dn2v_factory("../DN2V_SVC.conf")
# self.configs[dsname] = (dim, p, q, walk_len, num_walks, cumulative)

for ds in datasets:
    params = dn2v_fact.configs[ds]
    if params is not None:
        print("DATASET: ", ds)
        es, resol = load_event_sequence(ds)
        dg = DynamicGraph()
        dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
        graph_sequence = dg.get_graph_sequence()

        for i in range(0, 10):
            print()
            print("Iter: ", i)
            elid1 = DynN2V_LIDDelta(
                dim=params[0],
                p=params[1],
                q=params[2],
                walk_length=params[3],
                num_walks=params[4],
            )
            elid1.make_embedding(dg)
            emb1l = elid1.embeddings[-1]
            graphl1 = graph_sequence[-1]
            gre1 = GraphReconstructionEval(graphl1, emb1l)
            print("----DynN2V_LIDDelta----")
            print("Precision", gre1.get_avg_precision())
            print("Recall", gre1.get_avg_recall())
            print("F1", gre1.getF1())

            elid2 = DynN2V_LIDRW(
                dim=params[0],
                p=params[1],
                q=params[2],
                walk_length=params[3],
                num_walks=params[4],
            )
            elid2.make_embedding(dg)
            emb2l = elid2.embeddings[-1]
            graphl2 = graph_sequence[-1]
            gre2 = GraphReconstructionEval(graphl2, emb2l)
            print("----DynN2V_LIDRW----")
            print("Precision", gre2.get_avg_precision())
            print("Recall", gre2.get_avg_recall())
            print("F1", gre2.getF1())

            elid3 = DynN2V_LIDRWDelta(
                dim=params[0],
                p=params[1],
                q=params[2],
                walk_length=params[3],
                num_walks=params[4],
            )
            elid3.make_embedding(dg)
            emb3l = elid3.embeddings[-1]
            graphl3 = graph_sequence[-1]
            gre3 = GraphReconstructionEval(graphl3, emb3l)
            print("----DynN2V_LIDDelta----")
            print("Precision", gre3.get_avg_precision())
            print("Recall", gre3.get_avg_recall())
            print("F1", gre3.getF1())

    print("-----------------------------------------------")
    print()
    print()
