"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for tuning LID random-walk based dynamic graph embedding methods
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
Q = [0.5, 1.0, 2.0, 4.0]

NUM_RUNS = 10

WORKERS = 4
WALKLEN = 32
NUMWALKS = 10

from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicGraph

from tigra.embeddings.hypertuner import (
    DynN2V_LIDDelta_Tuner,
    DynN2V_LIDRW_Tuner,
    DynN2V_LIDRWDelta_Tuner,
)


def tune_DynN2V_LIDDelta(last_graph, dg):
    print("\nTUNING DynN2V_LIDDelta")
    tuner = DynN2V_LIDDelta_Tuner(
        last_graph,
        dg,
        P,
        Q,
        [WALKLEN],  # walk length
        [NUMWALKS],  # num walks
        num_runs=NUM_RUNS,
        parallel_processes=WORKERS,
    )
    tuner.tune()
    tuner.print_report()


def tune_DynN2V_LIDRW(last_graph, dg):
    print("\nTUNING DynN2V_LIDRW")
    tuner = DynN2V_LIDRW_Tuner(
        last_graph,
        dg,
        P,
        Q,
        [WALKLEN],  # walk length
        [NUMWALKS],  # num walks
        num_runs=NUM_RUNS,
        parallel_processes=WORKERS,
    )
    tuner.tune()
    tuner.print_report()


def tune_DynN2V_LIDRWDelta(last_graph, dg):
    print("\nTUNING DynN2V_LIDRWDelta")
    tuner = DynN2V_LIDRWDelta_Tuner(
        last_graph,
        dg,
        P,
        Q,
        [WALKLEN],  # walk length
        [NUMWALKS],  # num walks
        num_runs=NUM_RUNS,
        parallel_processes=WORKERS,
    )
    tuner.tune()
    tuner.print_report()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: rwtuner dataset-index")
        sys.exit()

    print("RWTUNER")
    print("P", P)
    print("Q", Q)
    print("NUM-RUNS", NUM_RUNS)
    print("WORKERS", WORKERS)
    print("WALKLEN", WALKLEN)
    print("NUMWALKS", NUMWALKS)

    dataset_index = int(sys.argv[1])
    dataset_name = datasets[dataset_index]
    print("\nTunning", dataset_name)

    es, time_resolution = load_event_sequence(dataset_name)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=time_resolution, directed=False)
    dg.info()

    org_graphs = dg.get_graph_sequence()
    last_graph = org_graphs[-1]

    start_time = time.time()
    tune_DynN2V_LIDRW(last_graph, dg)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("DynN2V_LIDRW TIME [sec]", elapsed_time)

    """
    start_time = time.time()
    tune_DynN2V_LIDDelta(last_graph, dg)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("DynN2V_LIDDelta TIME [sec]", elapsed_time)

    start_time = time.time()
    tune_DynN2V_LIDRWDelta(last_graph, dg)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("DynN2V_LIDRWDelta TIME [sec]", elapsed_time)
    """
