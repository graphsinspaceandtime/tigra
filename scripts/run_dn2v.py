"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for running node2vec dynamic graph embedding method
"""

import sys

sys.path.append("..")
sys.dont_write_bytecode = True

DATASET_INDEX = 8
DIM           = 50
P             = 0.5
Q             = 4.0
WORKERS       = 16
WALKLEN       = 64
NUMWALKS      = 50

from data.bank import Databank
databank = Databank()
datasets = databank.get_keys()

from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.core.greval import GraphReconstructionEval


def run_dn2v(dg):
    dn2v = Dynnode2vec(DIM, 
        p=P, 
        q=Q, 
        walk_length=WALKLEN, 
        n_walks_per_node=NUMWALKS, 
        parallel_processes=WORKERS
    )
    
    dn2v.make_embedding(dg)

    org_graphs = dg.get_graph_sequence()
    num_graphs = len(org_graphs)

    for i in range(num_graphs):
        og = org_graphs[i]
        emb = dn2v.embeddings[i]

        print("#nodes", og.num_nodes(), "#links", og.num_edges())
        num_links = og.num_edges()

        rg = emb.reconstruct(num_links, "REC")
        print("#nodes", rg.num_nodes(), "#links", rg.num_edges())


        gre = GraphReconstructionEval(og, emb, report_warnings=False)
        print("Snapshot", i)
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())
        print()




if __name__ == "__main__":
    dataset = datasets[DATASET_INDEX]

    print("Running dynnode2vec for", dataset)
    print("DIM", DIM)
    print("P", P)
    print("Q", Q)
    print("WORKERS", WORKERS)
    print("WALKLEN", WALKLEN)
    print("NUMWALKS", NUMWALKS)
    
    es, time_resolution = load_event_sequence(dataset)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=time_resolution, directed=False)
    dg.info()

    run_dn2v(dg)

