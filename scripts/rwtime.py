"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for measuring time performance of random-walk based dynamic graph embedding methods
"""

import sys
import os
import time
import psutil

WORKERS = 16
WALKLEN = 32
NUMWALKS = 10

def get_mem_usage_MB():
    process = psutil.Process()
    memory_usage = process.memory_info()[0] / float(2 ** 20)
    return memory_usage

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.embeddings.sota.ctdne_wrapper import Ctdne
from tigra.embeddings.sota.stwalk import STWalk

from data.bank import Databank
databank = Databank()
datasets = databank.get_keys()
print("DATASETS", datasets)

from tigra.embeddings.hypertuner import EMBEDDING_DIMENSIONS
print("EMBEDDING DIMENSIONS", EMBEDDING_DIMENSIONS)

def load_dataset(dataset_name):
    print("\nDATASET:", dataset_name)
    es = databank.get_data(dataset_name)
    ts_days = es.get_timespan_in_days()
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    
    resolution = "3-months"
    if ts_days < 10:
        resolution = "day"
    elif ts_days < 1000:
        resolution = "month"
    
    print("resolution=", resolution)
    return es, resolution


def form_temporal_graphs(dataset_name, es, resolution):
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resolution, directed=False)
    dg.info()

    org_graphs = dg.get_graph_sequence()
    last_graph = org_graphs[-1]

    dag = DynamicAggregateGraph(dataset_name, es, directed=False)

    return dg, last_graph, dag


def measure_dynnode2vec(dg, last_graph, dag):
    for d in EMBEDDING_DIMENSIONS:
        start_time = time.time()
        ok1 = True
        try:
            dn2v = Dynnode2vec(
                d, 
                p=1.0,
                q=1.0,
                walk_length=WALKLEN,
                n_walks_per_node=NUMWALKS,
                parallel_processes=WORKERS
            )
            dn2v.make_embedding(dg)
        except:
            ok1 = False
        end_time = time.time()
        t = round(end_time - start_time, 2)

        start_time = time.time()
        ok2 = True
        try:
            last_embedding = dn2v.embeddings[-1]
            GraphReconstructionEval(last_graph, last_embedding, report_warnings=False)
        except:
            ok2 = False
        end_time = time.time()
        rt = round(end_time - start_time, 2)

        tot = round(t + rt, 2)

        s = "DN2V," + str(d) + "," + str(t) + "," + str(rt) + "," + str(tot) + "," + str(ok1) + "," + str(ok2)
        print(s)


def measure_ctdne(dg, last_graph, dag):
    for d in EMBEDDING_DIMENSIONS:
        start_time = time.time()
        ok1 = True
        try:
            ctdne = Ctdne(
                d, 
                p=1.0, 
                q=1.0, 
                walk_length=WALKLEN,
                num_walks=NUMWALKS, 
                workers=WORKERS)
            ctdne.make_embedding(dag)
        except:
            ok1 = False
        end_time = time.time()
        t = round(end_time - start_time, 2)

        start_time = time.time()
        ok2 = True
        try:
            GraphReconstructionEval(dag.get_aggregate_graph(), ctdne.embeddings[0], report_warnings=False)
        except:
            ok2 = False
        end_time = time.time()
        rt = round(end_time - start_time, 2)

        tot = round(t + rt, 2)

        s = "CTDNE," + str(d) + "," + str(t) + "," + str(rt) + "," + str(tot) + "," + str(ok1) + "," + str(ok2)
        print(s)


def measure_stwalk(dg, last_graph, dag):
    for d in EMBEDDING_DIMENSIONS:
        start_time = time.time()
        ok1 = True
        try:
            ts = len(dg.graphs)
            st = STWalk(
                number_restart=NUMWALKS,
                walk_length=WALKLEN,
                dimension=d,
                time_step=ts,
                time_window_size=5,
                workers=WORKERS,
                vocab_window_size=5
            )
            st.make_embedding(dg)
        except:
            ok1 = False
        end_time = time.time()
        t = round(end_time - start_time, 2)

        start_time = time.time()
        ok2 = True
        try:
            last_embedding = st.embeddings[-1]
            GraphReconstructionEval(last_graph, last_embedding, report_warnings=False)
        except:
            ok2 = False
        end_time = time.time()
        rt = round(end_time - start_time, 2)

        tot = round(t + rt, 2)

        s = "STWALK," + str(d) + "," + str(t) + "," + str(rt) + "," + str(tot) + "," + str(ok1) + "," + str(ok2)
        print(s)


if __name__ == "__main__":
    for i in range(0, len(datasets) - 4):
        es, resol = load_dataset(datasets[i])
        dg, last_graph, dag = form_temporal_graphs(datasets[i], es, resol)
        mem = get_mem_usage_MB()
        print("MEMORY usage [MB]", mem)

        measure_dynnode2vec(dg, last_graph, dag)
        measure_ctdne(dg, last_graph, dag)
        measure_stwalk(dg, last_graph, dag)