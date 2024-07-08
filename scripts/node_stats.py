"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning#
in Complex Dynamical Systems

Script for node metrics calculation
"""

import sys

sys.path.append("..")
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from tigra.core.loaders import load_event_sequence
from networkx import centrality

from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

def __all_nodes(graph_seq):
    nodes = set()
    for g in graph_seq:
        for node in g.G.nodes:
            nodes.add(node)
    return nodes

def compute_and_write_metrics(graph_seq, filename):
    
    """calculates metrics for each node in each snapshot"""
    
    file_eigen = open(filename + "_eigenvector.csv", mode = "a",  encoding="utf-8")
    file_close = open(filename + "_closeness.csv", mode = "a",  encoding="utf-8")
    file_betw = open(filename + "_betweenness.csv", mode = "a",  encoding="utf-8")
    file_eigen.write("snap/node,")
    file_close.write("snap/node,")
    file_betw.write("snap/node,")
    
    allnodes = __all_nodes(graph_seq)
    for node in allnodes:
        file_eigen.write(str(node)+",")
        file_close.write(str(node)+",")
        file_betw.write(str(node)+",")
    file_eigen.write("\n")
    file_close.write("\n")
    file_betw.write("\n")
    
    snap = 0
    for gs in graph_seq:
        graph = gs.G
        
        eigenvector = dict(centrality.eigenvector_centrality(graph))
        closeness = dict(centrality.closeness_centrality(graph))
        betweenness = dict(centrality.betweenness_centrality(graph))
        
        file_eigen.write(str(snap) + ",")
        file_close.write(str(snap) + ",")
        file_betw.write(str(snap) + ",")
        
        for node in allnodes:
            if node in eigenvector.keys():
                file_eigen.write(str(eigenvector[node]) + ",")
                file_close.write(str(closeness[node]) + ",")
                file_betw.write(str(betweenness[node]) + ",")
            else:
                file_eigen.write("NaN,")
                file_close.write("NaN,")
                file_betw.write("NaN,")
        
        file_eigen.write("\n")
        file_close.write("\n")
        file_betw.write("\n")
        snap += 1
    
    file_eigen.close()
    file_close.close()
    file_betw.close()
        


if __name__ == "__main__":
    print(databank.get_keys())
    if (len(sys.argv) < 1):
        sys.exit()
    
    dataset_index = int(sys.argv[1])
    dataset_name = datasets[dataset_index]
   
    es, time_res = load_event_sequence(dataset_name)
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=time_res, directed=False)
    seq = dg.get_graph_sequence()
        
    compute_and_write_metrics(seq, "metrics_" + dataset_name)