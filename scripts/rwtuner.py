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

import data.bank as databank
datasets = databank.get_keys()

P = [0.5, 1.0, 2.0, 4.0]
#P = [1.0]

Q = [0.5, 1.0, 2.0, 4.0]
#Q = [1.0]

TIMEWINDOW = [1, 3, 5]

NUM_RUNS = 10
#NUM_RUNS = 1

WORKERS = 16
WALKLEN = 32
NUMWALKS = 10

from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph
from tigra.embeddings.hypertuner import Dynnode2vec_Tuner, CTDNE_Tuner, STWalk_Tuner


def get_resolution(dataset_name):
    if dataset_name in (datasets[0], datasets[1], datasets[2]):
        return "month"
    else:
        return "3-months"


def tune_dynnode2vec(last_graph, dg):
    print("\nTUNING dynnode2vec")
    tuner = Dynnode2vec_Tuner(
        last_graph, 
        dg, 
        P, 
        Q, 
        num_runs=NUM_RUNS, 
        parallel_processes=WORKERS, 
        walk_length=WALKLEN, 
        n_walks_per_node=NUMWALKS
    )
    tuner.tune()
    tuner.print_report()

def tune_ctdne(dag):
    print("\nTUNING CTDNE")
    tuner = CTDNE_Tuner(
        dag.get_aggregate_graph(),
        dag,
        P,
        Q,
        num_runs=NUM_RUNS,
        parallel_processes=WORKERS, 
        walk_length=WALKLEN, 
        n_walks_per_node=NUMWALKS
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


def load_dataset(dataset_name):
    print("DATASET:", dataset_name)
    es = databank.get_data(dataset_name)
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    return es


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("USAGE: rwtuner dataset-index")
        sys.exit()

    start_time = time.time()

    print("RWTUNER")
    print("P", P)
    print("Q", Q)
    print("NUM-RUNS", NUM_RUNS)
    print("WORKERS", WORKERS)
    print("WALKLEN", WALKLEN)
    print("NUMWALKS", NUMWALKS)

    dataset_index = int(sys.argv[1])
    dataset_name = datasets[dataset_index]
    es = load_dataset(dataset_name)
    time_resolution = get_resolution(dataset_name)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=time_resolution, directed=False)
    dg.info()

    org_graphs = dg.get_graph_sequence()
    last_graph = org_graphs[-1]

    dag = DynamicAggregateGraph(dataset_name, es, directed=False)

    tune_dynnode2vec(last_graph, dg)
    tune_ctdne(dag)
    tune_stwalk(last_graph, dg)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("TIME [sec]", elapsed_time)

    