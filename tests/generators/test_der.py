from tigra.generators.der import DERGenerator
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph

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

print()
print("Non-cumulative DER")
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False, cumulative=False)
dg.info()
for g in dg.get_graph_sequence():
    g.determine_connected_comps(to_print=True)



