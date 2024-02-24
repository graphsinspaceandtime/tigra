import sys

sys.path.append("..")
sys.dont_write_bytecode = True

from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
import data.bank as databank

es = databank.get_data("email-Eu-core-temporal")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False)
dg.info()

dn2v = Dynnode2vec(5)
dn2v.make_embedding(dg)

for emb in dn2v.embeddings:
    print("\n")
    print(emb)
