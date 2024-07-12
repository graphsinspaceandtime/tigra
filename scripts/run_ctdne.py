"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for running CTDNE dynamic graph embedding method
"""

import sys

sys.path.append("..")
sys.dont_write_bytecode = True

from data.bank import Databank
databank = Databank()
datasets = databank.get_keys()

from tigra.embeddings.sota.ctdne_wrapper import Ctdne
from tigra.core.dgraph import DynamicAggregateGraph, DynamicGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.core.loaders import load_event_sequence

from statistics import mean
import networkx as nx
import numpy as np

from scipy.stats import spearmanr, kendalltau


def run_ctdne(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words):
    ctdne = Ctdne(dim, 
        p=p, 
        q=q, 
        walk_length=walk_length, 
        num_walks=num_walks, 
        workers=workers, 
        window=window, 
        min_count=min_count, 
        batch_words=batch_words
    )

    ctdne.make_embedding(dag)

    gre = GraphReconstructionEval(dag.get_aggregate_graph(), ctdne.embeddings[0])
    print("Precision", gre.get_avg_precision())
    print("Recall", gre.get_avg_recall())
    print("F1", gre.getF1())

    return ctdne.embeddings[0]


def eval_weighted(emb, dag):
    G = dag.get_aggregate_graph().G
    edges = G.edges()
    x, y = list(), list()
    for e in edges:
        src_node = e[0]
        dst_node = e[1]
        w = G.edges[e]["weight"]
        v1 = np.array(emb[src_node])
        v2 = np.array(emb[dst_node])
        dist = np.linalg.norm(v1 - v2)
        x.append(dist)
        y.append(w)

    sp = spearmanr(x, y)
    print("Spearman correlation", sp)
    kt = kendalltau(x, y)
    print("Kendal Tau correlation", kt)




def examine_num_walks(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words):
    num_walks_options = [num_walks, num_walks * 2, num_walks * 3]
    
    for nw in num_walks_options:
        print("NUM-WALKS", nw)
        run_ctdne(dag, dim, p, q, walk_length, nw, workers, window, min_count, batch_words)

    
def examine_walk_length(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words):
    walk_length_options = [walk_length, walk_length * 2, walk_length * 3]
    
    for wl in walk_length_options:
        print("WALK-LENGTH", wl)
        run_ctdne(dag, dim, p, q, wl, num_walks, workers, window, min_count, batch_words)


def examine_window(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words):
    window_options = [window, window * 2, window * 3, window * 4, window * 5, window * 20]
    
    for w in window_options:
        print("WINDOW", w)
        run_ctdne(dag, dim, p, q, walk_length, num_walks, workers, w, min_count, batch_words)


def examine_min_count(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words):
    mc_options = [min_count, min_count * 2, min_count * 3, min_count * 5, min_count * 10]
    
    for mc in mc_options:
        print("MIN-COUNT", mc)
        run_ctdne(dag, dim, p, q, walk_length, num_walks, workers, window, mc, batch_words)


def examine_batch_words(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words):
    bw_options = [batch_words, batch_words * 2, batch_words * 3, batch_words * 5, batch_words * 10]
    
    for bw in bw_options:
        print("BATCH WORDS", bw)
        run_ctdne(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, bw)


def analyze_dag(G):
    print("Analyzing DAG")
    
    num_nodes = len(G.nodes())
    print("#nodes", num_nodes)

    num_links = len(G.edges())
    print("#links", num_links)

    is_weighted = nx.is_weighted(G, edge=None, weight='weight')
    print("Weighted?", is_weighted)
    if is_weighted:
        weights = nx.get_edge_attributes(G, 'weight')
        wlist = []
        for l in weights:
            wlist.append(weights[l])

        print("Average weight", mean(wlist))
        print("Max weight", max(wlist))

    comps = list(nx.connected_components(G))
    print("Connected components", len(comps))
    c_size = [len(c) for c in comps]
    c_size.sort(reverse=True)
    l_frac = c_size[0] / num_nodes
    print("Largest CC", l_frac)




if __name__ == "__main__":
    import pandas as pd
    conf = pd.read_csv("run_ctdne.conf")
    print(conf)

    dataset = str(conf[conf["PROPERTY"]=="DATASET"]["VALUE"].item())
    dim = int(conf[conf["PROPERTY"]=="DIM"]["VALUE"].item())
    p = float(conf[conf["PROPERTY"]=="P"]["VALUE"].item())
    q = float(conf[conf["PROPERTY"]=="Q"]["VALUE"].item())
    walk_length = int(conf[conf["PROPERTY"]=="WALK_LENGTH"]["VALUE"].item())
    num_walks = int(conf[conf["PROPERTY"]=="NUM_WALKS"]["VALUE"].item())
    workers = int(conf[conf["PROPERTY"]=="WORKERS"]["VALUE"].item())
    window = int(conf[conf["PROPERTY"]=="WINDOW"]["VALUE"].item())
    min_count = int(conf[conf["PROPERTY"]=="MIN_COUNT"]["VALUE"].item())
    batch_words = int(conf[conf["PROPERTY"]=="BATCH_WORDS"]["VALUE"].item())

    print(dataset, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words)

    es, time_resolution = load_event_sequence(dataset)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=time_resolution, directed=False)
    dg.info()

    print("BASELINE")
    dag = DynamicAggregateGraph(dataset, es, directed=False)
    run_ctdne(dag, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words) 

    print("\nWEIGHTED")
    dag2 = DynamicAggregateGraph(dataset, es, directed=False, weighted=True)
    emb = run_ctdne(dag2, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words)
    G = dag2.get_aggregate_graph().G
    analyze_dag(G)
    eval_weighted(emb, dag2)

    #examine_num_walks(dag2, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words)
    #examine_walk_length(dag2, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words)
    #examine_window(dag2, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words)
    #examine_min_count(dag2, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words)
    #examine_batch_words(dag2, dim, p, q, walk_length, num_walks, workers, window, min_count, batch_words)
