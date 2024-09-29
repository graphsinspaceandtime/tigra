import sys
sys.path.append("../..")

from tigra.generators.der import DERGenerator
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.core.dnea import EvolDynamics

from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.core.greval import GraphReconstructionEval

#derg = DERGenerator(1000, 0.01, time_resolution="hour")
#derg.generate_and_save("der_1000_001.events")

es = load_snap_dataset("der_1000_001.events")

print("Cumulative DER")
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False, cumulative=True)
dg.info()
last_snap = dg.get_graph_sequence()[-1]
print("Degree distribution")
last_snap.degree_distribution(to_print=True)
for g in dg.get_graph_sequence():
    g.determine_connected_comps(to_print=True)
ed = EvolDynamics(dg)
ed.info()

print("########################")

dn2v = Dynnode2vec(100)
dn2v.make_embedding(dg)

org_graphs = dg.get_graph_sequence()

last_embedding = dn2v.embeddings[-1]
last_graph = org_graphs[-1]
print("Original graph", last_graph.num_nodes(), last_graph.num_edges())
print("#emb-vectors", len(last_embedding.embedding.keys()))

gre = GraphReconstructionEval(last_graph, last_embedding, report_warnings=False)
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())

"""
print()
print("Non-cumulative DER")
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False, cumulative=False)
dg.info()
for g in dg.get_graph_sequence():
    g.determine_connected_comps(to_print=True)
ed = EvolDynamics(dg)
ed.info()
"""

