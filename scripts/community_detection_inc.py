import sys
import os

sys.path.append("..")
sys.path.append(os.path.abspath("."))

import tigra
import networkx as nx
from community import community_louvain
from copy import deepcopy
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score
from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.loaders import load_event_sequence


def _patch_number_of_edges(cls):
    if not hasattr(cls, "number_of_edges") and hasattr(cls, "size"):
        cls.number_of_edges = cls.size
    if not hasattr(cls, "number_of_edges") and hasattr(cls, "num_edges"):
        cls.number_of_edges = cls.num_edges


_patch_number_of_edges(nx.Graph)
_patch_number_of_edges(tigra.core.dgraph.Graph)
_patch_number_of_edges(nx.DiGraph)

databank = Databank()
datasets = databank.get_keys()


def incremental_louvain(graph_snapshots, delta_threshold=0):
    communities_over_time = []

    G_prev = graph_snapshots[0].G
    C_prev = community_louvain.best_partition(G_prev, weight="weight")
    communities_over_time.append(C_prev)

    for t in range(1, len(graph_snapshots)):
        G_curr = graph_snapshots[t].G

        edges_prev = set(G_prev.edges())
        edges_curr = set(G_curr.edges())
        delta_added = edges_curr - edges_prev
        delta_removed = edges_prev - edges_curr
        delta_total = len(delta_added) + len(delta_removed)

        if delta_total <= delta_threshold:
            C_curr = deepcopy(C_prev)
        else:
            C_curr = update_louvain(G_curr, C_prev, delta_added, delta_removed)

        communities_over_time.append(C_curr)
        G_prev, C_prev = G_curr, C_curr

    return communities_over_time


def update_louvain(G_curr, C_prev, delta_added, delta_removed):
    C_new = deepcopy(C_prev)

    affected_nodes = set()
    for u, v in delta_added.union(delta_removed):
        affected_nodes.add(u)
        affected_nodes.add(v)

    if affected_nodes:
        sub_nodes = set(affected_nodes)
        for n in affected_nodes:
            if n not in G_curr:
                continue
            sub_nodes.update(G_curr.neighbors(n))

        subG = G_curr.subgraph(sub_nodes).copy()
        local_partition = community_louvain.best_partition(subG, weight="weight")

        for n, comm in local_partition.items():
            C_new[n] = comm

    return C_new


def dict_to_communities(communities_dict):
    comm_sets = {}
    for node, comm in communities_dict.items():
        comm_sets.setdefault(comm, set()).add(node)
    return list(comm_sets.values())


def evaluate_modularity(graph_snapshots, communities_over_time):
    modularities = []
    num_communities = []
    nmi_scores = []
    C_prev = None

    for t in range(1, len(graph_snapshots) + 1):
        G_t = graph_snapshots[t - 1]
        C_t = communities_over_time[t - 1]
        
        G = G_t.G

        comm_sets = {}
        for node, comm in C_t.items():
            if node in G:
                comm_sets.setdefault(comm, set()).add(node)

        comm_list = list(comm_sets.values())

        all_nodes_in_partition = set().union(*comm_list) if comm_list else set()
        missing_nodes = set(G.nodes()) - all_nodes_in_partition
        for n in missing_nodes:
            comm_list.append({n})

        Q_t = modularity(G, comm_list, weight="weight")
        modularities.append(Q_t)

        num_comms = len(comm_list)
        num_communities.append(num_comms)

        if C_prev is not None:
            prev_labels = [C_prev.get(n, -1) for n in G.nodes()]
            curr_labels = [C_t.get(n, -1) for n in G.nodes()]
            nmi = normalized_mutual_info_score(prev_labels, curr_labels)
            nmi_scores.append(nmi)
            print(f"Snapshot {t}: Q = {Q_t:.4f}, NC = {num_comms}, NMI = {nmi:.4f}")
        else:
            nmi_scores.append(None)
            print(f"Snapshot {t}: Q = {Q_t:.4f}, NC = {num_comms}")

        C_prev = C_t

    return modularities, num_communities, nmi_scores


names = [
    "CollegeMsg",
    "email-Eu-core-temporal",
    "fb-forum",
    "fb-messages",
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    "ia-primary-school-proximity-attr",
    # "ia-reality-call",
    "radoslaw-email",
]


def load_novel_emb(dataset):
    es, resol = load_event_sequence(dataset)
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()
    return dg


if __name__ == "__main__":
    os.makedirs("community_detection/plot", exist_ok=True)

    for dataset in names:
        print(dataset)
        dg = load_novel_emb(dataset)
        graph_sequence = dg.get_graph_sequence()

        communities_over_time = incremental_louvain(graph_sequence, delta_threshold=0)
        modularities, num_communities, nmi_scores = evaluate_modularity(
            graph_sequence, communities_over_time
        )

        path = os.path.join("community_detection", f"results_{dataset}.csv")
        with open(path, "w") as f:
            f.write("snapshot,Q,NC,NMI\n")
            
            for t in range(len(modularities)):
                
                q = modularities[t]
                nc = num_communities[t]
                nmi = nmi_scores[t]
                
                f.write(f"{t},{q},{nc},{nmi}\n")
