import sys
from datetime import timedelta

sys.path.append("..")
sys.dont_write_bytecode = True

from tigra.core.loaders import load_netrep_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.sota.dynnode2vec import Dynnode2vec

es = load_netrep_dataset("../data/netrep/ia-radoslaw-email.edges")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution=timedelta(days=31), directed=False)
dg.info()

dn2v = Dynnode2vec(5)
dn2v.make_embedding(dg)

for emb in dn2v.embeddings:
    print("\n")
    print(emb)
