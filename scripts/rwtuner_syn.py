import sys

from rwtuner import tune_dynnode2vec, tune_ctdne, tune_stwalk
from rwtuner import P, Q, NUM_RUNS, NUMWALKS, WORKERS, WALKLEN, TIMEWINDOW
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph

import networkx as nx

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("USAGE: rwtuner dataset-name time-resolution")
        sys.exit()

    print("RWTUNER")
    print("P", P)
    print("Q", Q)
    print("NUM-RUNS", NUM_RUNS)
    print("WORKERS", WORKERS)
    print("WALKLEN", WALKLEN)
    print("NUMWALKS", NUMWALKS)
    print("TIMEWINDOW", TIMEWINDOW)
    
    dataset_name = sys.argv[1]
    timeres = sys.argv[2]
    print(dataset_name, timeres)
    
    es = load_snap_dataset(dataset_name)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=timeres, directed=False, cumulative=True)
    dg.info()
    
    org_graphs = dg.get_graph_sequence()
    last_graph = org_graphs[-1]
    
    agr_graph = DynamicAggregateGraph(dataset_name, es, directed=False)
    print(agr_graph.agg_graph.num_nodes())
    print(agr_graph.agg_graph.num_edges())
    
    #print("TUNING DYNNODE2VEC")
    #tune_dynnode2vec(last_graph, dg)
    #print()
    #print("TUNING CTDNE")
    #tune_ctdne(last_graph, agr_graph)
    
    
    print("DEBUG........")
    seq = dg.get_nx_undirected_graph_sequence()
    for g in seq:
        print(list(nx.isolates(g)))
        
    print("END DEBUG......")
    
    print()
    print("TUNING STWALK")
    tune_stwalk(last_graph, dg)
    

