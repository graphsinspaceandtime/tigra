"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for lid_eval tests
"""

import sys
import os

sys.path.append("..")
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from tigra.evaluation.lid_eval import LFMnx
from tigra.evaluation.lid_eval import NCLIDEstimator

from numpy import std
from scipy.stats import skew

from data.bank import Databank
databank = Databank()
datasets = databank.get_keys()


def compute_expansion(node, community, graph):
    no_good_links = 0
    no_bad_links = 0
    set_comm = set(community)
    for node in set_comm:
        neigh = graph.neighbors(node)
        for n in neigh:
            if n in set_comm:
                no_good_links += 1
            else:
                no_bad_links += 1
                
    no_good_links /= 2.0
    
    return float(no_bad_links)/float(no_good_links + no_bad_links)

def __all_nodes(seq):
    nodes = set()
    for g in seq:
        for node in g.G.nodes:
            nodes.add(node)
    return nodes

def compute_and_write_lids(seq, file_name):
    nc_lids = dict()
    snap = 0

    for g in seq:
        estimator = NCLIDEstimator(g.G)
        nc_lids[snap] = dict()
        estimator.estimate_lids()
        lid_vals = estimator.get_lid_values()
    
        for node in g.G.nodes:
            nc_lids[snap][node] = lid_vals.get(node)
        
        snap +=1
    
    f = open(file_name, "a")
    nodes = __all_nodes(seq)
    f.write("snapshot/node,")
    for node in nodes:
        f.write(str(node)+",")
    f.write("\n")
    
    for s in range(snap):
        f.write(str(s) + ",")
        for node in nodes:
            if node in nc_lids[s].keys():
                f.write(str(nc_lids[s][node]) + ",")
            else:
                f.write("NaN,")
        f.write("\n")
        
    f.close()
    


if __name__ == "__main__":
    if (len(sys.argv) < 1):
        sys.exit()
    
    dataset_index = int(sys.argv[1])
    dataset_name = datasets[dataset_index]
    es = databank.get_data(dataset_name)
    
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution="month", directed=False)
    
    seq = dg.get_graph_sequence()
    
    compute_and_write_lids(seq, "nc_lid_" + dataset_name + ".csv")
    