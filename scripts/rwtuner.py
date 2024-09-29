"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for tuning random-walk based dynamic graph embedding methods
"""

import sys
import os
import time

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from data.bank import Databank
databank = Databank()
datasets = databank.get_keys()

P = [0.5, 1.0, 2.0, 4.0]
#P = [1.0]

Q = [0.5, 1.0, 2.0, 4.0]
#Q = [1.0]

TIMEWINDOW = [1, 3, 5]

NUM_RUNS = 5
#NUM_RUNS = 1

WORKERS = 16
WALKLEN = 32
NUMWALKS = 10
#NUMWALKS = 1

from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph
from tigra.embeddings.hypertuner import Dynnode2vec_Tuner, CTDNE_Tuner, STWalk_Tuner

"""
# TODO refactoring, see load_dataset from rwtime.py
def get_resolution(dataset_name):
    if dataset_name in (datasets[0], datasets[1], datasets[2], datasets[7], datasets[8], datasets[9]):
        return "month"
    elif dataset_name in (datasets[3], datasets[5], datasets[6]):
        return "day"
    else:
        return "3-months"
"""

def tune_dynnode2vec(last_graph, dg):
    print("\nTUNING dynnode2vec")
    tuner = Dynnode2vec_Tuner(
        last_graph, 
        dg, 
        P, 
        Q,
        [WALKLEN],                  # walk length
        [NUMWALKS],                  # num walks
        num_runs=NUM_RUNS, 
        parallel_processes=WORKERS
    )
    tuner.tune()
    tuner.print_report()


def tune_ctdne(last_graph, dag):
    print("\nTUNING CTDNE")
    tuner = CTDNE_Tuner(
        dag.get_aggregate_graph(),
        dag,
        P,
        Q,
        [WALKLEN],                              # walk length
        [NUMWALKS],                           # num walks
        num_runs=NUM_RUNS,
        parallel_processes=WORKERS
    )
    tuner.tune()
    tuner.print_report()


def tune_stwalk(last_graph, dg):
    print("\nTUNING STWALK")
    tuner = STWalk_Tuner(
        last_graph, 
        dg, 
        TIMEWINDOW,
        num_runs=NUM_RUNS,
        parallel_processes=WORKERS, 
        walk_length=WALKLEN, 
        n_walks_per_node=NUMWALKS
    )
    tuner.tune()
    tuner.print_report()


"""
def load_dataset(dataset_name):
    print("DATASET:", dataset_name)
    es = databank.get_data(dataset_name)
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    return es
"""

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("USAGE: rwtuner dataset-index")
        sys.exit()

    print("RWTUNER")
    print("P", P)
    print("Q", Q)
    print("NUM-RUNS", NUM_RUNS)
    print("WORKERS", WORKERS)
    print("WALKLEN", WALKLEN)
    print("NUMWALKS", NUMWALKS)
    print("TIMEWINDOW", TIMEWINDOW)

    dataset_index = int(sys.argv[1])
    dataset_name = datasets[dataset_index]
    print("\nTunning", dataset_name)

    #es = load_dataset(dataset_name)
    #time_resolution = get_resolution(dataset_name)
    #print("TIME-RESOLUTION", time_resolution)

    es, time_resolution = load_event_sequence(dataset_name)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=time_resolution, directed=False)
    dg.info()

    org_graphs = dg.get_graph_sequence()
    last_graph = org_graphs[-1]

    #dag = DynamicAggregateGraph(dataset_name, es, directed=False)

    start_time = time.time()
    tune_dynnode2vec(last_graph, dg)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("DYNNODE2VEC TIME [sec]", elapsed_time)
    
    """
    start_time = time.time()
    tune_ctdne(last_graph, dag)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("CTDNE TIME [sec]", elapsed_time)
    
    
    start_time = time.time()
    tune_stwalk(last_graph, dg)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("STWALK TIME [sec]", elapsed_time)
    """