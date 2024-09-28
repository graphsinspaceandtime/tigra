from tigra.generators.dba import DBAGenerator
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph

#derg = DBAGenerator(1000, 5, 10, time_resolution="hour")
#derg.generate_and_save("dba_1000_5_10.events")

es = load_snap_dataset("dba_1000_5_10.events")

print("Cumulative DBA")
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False, cumulative=True)
dg.info()
last_snap = dg.get_graph_sequence()[-1]
print("Degree distribution")
last_snap.degree_distribution(to_print=True)
for g in dg.get_graph_sequence():
    g.determine_connected_comps(to_print=True)

print()
print("Non-cumulative DBA")
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False, cumulative=False)
dg.info()
for g in dg.get_graph_sequence():
    g.determine_connected_comps(to_print=True)