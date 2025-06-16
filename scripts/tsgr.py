import sys

sys.path.append("..")

import os
import networkx as nx
import pandas as pd

from tigra.core.dgraph import DynamicGraph, Graph

def load_data(directory, prefix):
    matrices = {}
    
    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname.lower().endswith('.csv'):
            full_path = os.path.join(directory, fname)
            
            df = pd.read_csv(full_path, header=None)
            matrix = df.values
            matrices[fname] = matrix
    
    return matrices

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
        count+=1
        if(nx.number_connected_components(graph) == 1):
            print(count)
            return dist
        
        
def tsgr(snapshots, grt):
    graps = []
    for i, mat in snapshots.items():
        ordered_nodes = order_nodes_by_distance(mat)
        graph = nx.Graph()
        graph.add_nodes_from(range(mat.shape[0]))
        for pair in ordered_nodes:
            if pair[2] < grt:
                graph.add_edge(pair[0], pair[1])
        graps.append(Graph(f"snapshot {i}", nx_graph=graph))
    
    return DynamicGraph(list_of_graphs = graps)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "Plane_50_5_1.0")
    prefix_string = "Plane_50_5" 
    
    loaded = load_data(data_dir, prefix_string)
    distance_tresholds = set()

    for fname, mat in loaded.items():
        print(f"Loaded data from {fname}: {mat.shape}")
        distances = order_nodes_by_distance(mat)
        treshold = find_distance_treshold(distances, mat.shape[0])
        distance_tresholds.add(treshold)
        print(f"Snap number: {len(distance_tresholds)}, treshold: {treshold}")

    grt = max(distance_tresholds)
    print(f"Max distance {grt}")
    
    dyn_graph = tsgr(loaded, grt)
    dyn_graph.info()