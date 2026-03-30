import sys, os

print(sys.path)
sys.path.append("..")
sys.path.append(os.path.abspath("."))


import os
import networkx as nx
import pandas as pd
import pickle
import numpy as np


from tigra.core.dgraph import DynamicGraph, Graph


def load_data(directory, prefix):
    matrices = {}

    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname.lower().endswith(".csv"):
            full_path = os.path.join(directory, fname)

            df = pd.read_csv(full_path, header=None)
            matrix = df.values
            matrices[fname] = matrix

    return matrices


def load_labels(directory, labels):
    values = []

    for fname in os.listdir(directory):
        if fname == labels:
            full_path = os.path.join(directory, fname)
            values = pd.read_csv(full_path, header=None)[0].tolist()

    return values


def order_nodes_by_distance(matrix):
    distances = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            distance = matrix[i, j]
            distances.append((i, j, distance))

    distances.sort(key=lambda x: x[2])
    return distances


def find_distance_treshold(sorted_distances_list, num_nodes):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    count = 0
    for x in sorted_distances_list:
        src = x[0]
        goal = x[1]
        dist = x[2]
        graph.add_edge(src, goal)
        count += 1
        if nx.number_connected_components(graph) == 1:
            print(count)
            return dist


def tsgr(snapshots, grt, labels):
    graphs = []
    for i, mat in snapshots.items():
        ordered_nodes = order_nodes_by_distance(mat)
        graph = nx.Graph()
        graph.add_nodes_from(range(mat.shape[0]))
        for node_id, label in enumerate(labels):
            graph.nodes[node_id]["node_label"] = label
        for pair in ordered_nodes:
            if pair[2] < grt:
                graph.add_edge(pair[0], pair[1])
        graphs.append(Graph(f"snapshot {i}", nx_graph=graph, labeled=True))

    return DynamicGraph(list_of_graphs=graphs)


def tsgr_knn(snapshots, k, labels):
    graphs = []
    for i, mat in snapshots.items():
        n_nodes = mat.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))
        for node_id, label in enumerate(labels):
            graph.nodes[node_id]["node_label"] = label

        for node in range(n_nodes):
            neighbors = sorted(range(n_nodes), key=lambda other: mat[node][other])
            for neighbor in neighbors[1 : k + 1]:
                graph.add_edge(node, neighbor)

        graphs.append(Graph(f"snapshot {i}", nx_graph=graph, labeled=True))

    return DynamicGraph(list_of_graphs=graphs)


def load_tigra_dgraph(fname):
    with open(fname, "rb") as f:
        tigra_graphs = []
        graphs = pickle.load(f)
        for i, graph in enumerate(graphs):
            tigra_graphs.append(Graph(f"snapshot {i}", nx_graph=graph, labeled=True))
        dyn = DynamicGraph(list_of_graphs=tigra_graphs)
        dyn.info()

        return dyn


# expected to have dir Plane_50_5_1.0 with list of data files and labels file in the TIGRA root in case of running as standalone script from TIGRA/scripts
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "ArrowHead_50_5_1.0")
    prefix_string = "ArrowHead_50_5_1.0"

    labels = load_labels(data_dir, "ArrowHead.csv")

    loaded = load_data(data_dir, prefix_string)
    distance_tresholds = set()

    # max distance

    # for fname, mat in loaded.items():
    #     print(f"Loaded data from {fname}: {mat.shape}")
    #     distances = order_nodes_by_distance(mat)
    #     treshold = find_distance_treshold(distances, mat.shape[0])
    #     distance_tresholds.add(treshold)
    #     print(f"Snap number: {len(distance_tresholds)}, treshold: {treshold}")

    # grt = max(distance_tresholds)
    # print(f"Max distance {grt}")

    # dyn_graph = tsgr(loaded, grt, labels)

    # knn
    first_mat = next(iter(loaded.values()))
    n_nodes = first_mat.shape[0]
    k = int(np.ceil(np.log2(n_nodes)))
    print(f"K = {k}")

    dyn_graph = tsgr_knn(loaded, k, labels)
    dyn_graph.info()

    with open("arrow_head__knn_dynamic_graph.pkl", "wb") as f:
        pickle.dump(dyn_graph.get_nx_graph_sequence(), f)
