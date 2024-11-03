import sys

sys.path.append("../..")

from tigra.generators.dba import DBAGenerator, DBACGenerator
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.core.dnea import EvolDynamics

from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.core.greval import GraphReconstructionEval


def radicchi_in_out(community, G):
    total_in, total_out = 0, 0

    for node in community:
        if not node in G.nodes():
            continue

        neis = G.neighbors(node)
        links_in, links_out = 0, 0
        for nei in neis:
            if nei in community:
                links_in += 1
            else:
                links_out += 1

        total_in += links_in
        total_out += links_out

    return total_in / 2, total_out


def examine_partition(G, partition):
    print("# communities", len(partition))
    s = "["
    for i in range(len(partition)):
        s += str(len(partition[i]))
        if i < len(partition) - 1:
            s += ","
    s += "]"
    print("comm size", s)

    for i in range(len(partition)):
        total_in, total_out = radicchi_in_out(partition[i], G)
        print("Community", i, ", in = ", total_in, ", out = ", total_out)


# derg = DBAGenerator(1000, 5, 10, time_resolution="hour")
# derg.generate_and_save("dba_1000_5_10.events")

# dbac = DBACGenerator([500, 250, 150, 100], 4, 1, 8)
# dbac.generate_and_save("dbac_1000_4.events")

es = load_snap_dataset("dbac_1000_4.events")

print("Cumulative DBAC")
dg = DynamicGraph()
dg.init_from_event_sequence(
    es, time_resolution="month", directed=False, cumulative=True
)
dg.info()
last_snap = dg.get_graph_sequence()[-1]
print("Degree distribution")
last_snap.degree_distribution(to_print=True)
for g in dg.get_graph_sequence():
    g.determine_connected_comps(to_print=True)
ed = EvolDynamics(dg)
ed.info()

print("\nCommunity examination")
examine_partition(
    last_snap.G,
    [
        set(range(0, 500)),
        set(range(500, 750)),
        set(range(750, 900)),
        set(range(900, 1000)),
    ],
)

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
print("Non-cumulative DBA")
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False, cumulative=False)
dg.info()
for g in dg.get_graph_sequence():
    g.determine_connected_comps(to_print=True)
ed = EvolDynamics(dg)
ed.info()
"""
