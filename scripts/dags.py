"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script to investigate properties of dynamic aggregate graphs
"""

import sys

sys.path.append("..")
sys.dont_write_bytecode = True

import networkx as nx

from data.bank import Databank
databank = Databank()
datasets = databank.get_keys()

from tigra.core.dgraph import DynamicAggregateGraph
from tigra.core.loaders import load_event_sequence, datasets

def analyze_dag(dag):
    print("Analyzing DAG")
    
    G = dag.get_aggregate_graph().G

    num_nodes = len(G.nodes())
    print("#nodes", num_nodes)

    num_links = len(G.edges())
    print("#links", num_links)

    comps = list(nx.connected_components(G))
    print("Connected components", len(comps))
    c_size = [len(c) for c in comps]
    c_size.sort(reverse=True)
    l_frac = c_size[0] / num_nodes
    print("Largest CC", l_frac)
    
    avg_wl, max_wl, std_wl = dag.estimate_avg_temp_path_length()
    print(avg_wl, max_wl, std_wl)

    return str(num_nodes) + "," +\
        str(num_links) + "," +\
        str(len(comps)) + "," +\
        str(round(l_frac, 4)) + "," +\
        str(round(avg_wl, 4)) + "," +\
        str(round(max_wl, 4)) + "," +\
        str(round(std_wl, 4))


    
def examine_dag(dataset):
    print(dataset)
    es, _ = load_event_sequence(dataset)
    print("#events", es.num_events())

    dag = DynamicAggregateGraph(dataset, es, directed=False)
    return analyze_dag(dag)



if __name__ == "__main__":
    msg = []
    for dataset in datasets:
        es, resol = load_event_sequence(dataset)
        m = examine_dag(dataset)
        msg.append(dataset + "," + m)

    for m in msg:
        print(m)

    
