import sys
import random
import numpy as np
import networkx as nx

sys.path.append("../..")

from tigra.generators.dsbm import DSBMGenerator, DPPGenerator
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.core.dnea import EvolDynamics

from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.core.greval import GraphReconstructionEval

"""
$ C $ is a 2d array and $ c^i $ is a vector which corresponds to a row $ i $ in 2d array $ C $.
$ c $ is a vector containing class enumerations for each node in a graph. $ c_j $ is the class of
the $ j $-th node.

Each row $ i $ in $ C $ represents class memberships at time $ i $ in the dynamic graph.
"""
#              one of 3 classes         50 nodes            create 5 snapshots
#C = np.array([[random.choice([0, 1, 2]) for _ in range(50)] for _ in range(5)])

"""
$ P $ is a 3d array and $ P^i $ is a 2d array representing the matrix of probabilities for inter and
intra links between each class in graph at time $ i $.
"""
#               probabilities                                     create 5 snapshots
#P = np.array([[[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]] for _ in range(5)])

"""
gen = DSBMGenerator(C, P)

sequence = gen.generate()

print(
    "\n".join(
        [
            f"{event.src} - {event.dst} - {event.timestamp}"
            for event in sequence.get_events()
        ]
    )
)
"""

"""
def label_propagation(G):
    partition = nx.community.label_propagation_communities(G)
    examine_partition(G, partition)


def louvain(G):
    partition = nx.community.louvain_communities(G)
    examine_partition(G, partition)


def gmo(G):
    partition = nx.community.greedy_modularity_communities(G)
    examine_partition(G, partition) 
"""

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




#dppg = DPPGenerator([500, 250, 150, 100], 0.03, 0.001)
#dppg.generate_and_save('dpp_1000_4.events')

es = load_snap_dataset('dpp_1000_4.events')

print("Cumulative DPP")
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

print("\nCommunity examination")
examine_partition(last_snap.G, 
    [
        set(range(0, 500)), 
        set(range(500, 750)),
        set(range(750, 900)),
        set(range(900, 1000))
    ]
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
