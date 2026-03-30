import sys
import os
import networkx as nx
import numpy as np
import random
import time

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.greval import GraphReconstructionEval


from tigra.embeddings.rndw.rlwalk import DynRLWalk


if __name__ == "__main__":
    DATASET = "radoslaw-email"
    TIME_RES = "month"

    print(f"--- Loading {DATASET} ---")
    databank = Databank()
    es = databank.get_data(DATASET)
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
    dg.info()

    org_graphs = dg.get_graph_sequence()
    final_graph_wrapper = org_graphs[-1]


    dimensions_grid = [10, 25, 50, 100, 200]
    
    rl_params_grid = [
        {"eps": 0.1, "alpha": 0.1, "bias": False}, 
        {"eps": 0.3, "alpha": 0.1, "bias": False}, 
        {"eps": 0.2, "alpha": 0.1, "bias": True}, 
    ]


    print(f"{'DIM':<4} {'EPS':<5} {'ALPHA':<6} {'BIAS':<6} | {'PREC':<8} {'REC':<8} {'F1':<8}")


    best_f1 = -1.0
    best_config = {}


    for dim in dimensions_grid:
        
        for params in rl_params_grid:
            

            rl_model = DynRLWalk(
                dim=dim,  
                epsilon=params["eps"], 
                alpha=params["alpha"], 
                reward_bias=params["bias"],
                num_walks=10, 
                walk_length=40, 
                num_workers=1
            )
            
            try:
                rl_embeddings = rl_model.construct(dg)
                last_emb_rl = rl_embeddings[-1]


                gre_rl = GraphReconstructionEval(final_graph_wrapper, last_emb_rl)
                
                p = gre_rl.get_avg_precision()
                r = gre_rl.get_avg_recall()
                f1 = gre_rl.getF1()
                
                print(f"{dim:<4} {params['eps']:<5} {params['alpha']:<6} {str(params['bias']):<6} | {p:.4f}   {r:.4f}   {f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = {"dim": dim, **params}
                    
            except Exception as e:
                print(f"{dim:<4} {params['eps']:<5} ERROR: {e}")

    print("-" * 75)
    print("WINNER CONFIGURATION:")
    print(best_config)
    print(f"Best F1: {best_f1:.4f}")