import sys

sys.path.append("..")
sys.dont_write_bytecode = True

import numpy as np
from tigra.core.dgraph import DynamicGraph
from tigra.core.loaders import load_event_sequence
from tigra.embeddings.embedding import FSEmbedding
from tigra.core.greval import GraphReconstructionEval
from tigra.evaluation.diffusion import DiffusionPrediction
from tigra.evaluation.diffusion import SI

datasets = [
    "radoslaw-email",
    "fb-forum",
    "fb-messages",
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    #"ia-primary-school-proximity-attr",
    #"ia-reality-call",
    "CollegeMsg",
    "email-Eu-core-temporal",
]

def greval(dg, emb):
    graph_sequence = dg.get_graph_sequence()
    f1_list = []
    for i in range(len(graph_sequence)):
        emb_i = emb.embeddings[i]
        graph = graph_sequence[i]
        gre = GraphReconstructionEval(graph, emb_i)
        f1 = gre.getF1()
        f1_list.append(f1)

    return np.mean(f1_list)

def get_embeddings(ds_name):
    ds_name_2 = ds_name.replace("-", "_").lower()
    if ds_name == "ia-contacts_hypertext2009":
        ds_name_2 = "ia_contacts_hypertext_2009"
    if ds_name == "CollegeMsg":
        ds_name_2 = "college_msg"

    es, resol = load_event_sequence(ds_name)
    #print("num-events=", es.num_events())
    #print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    #dg.info()

    emb_dn2v = FSEmbedding(f"../../tigra_emb/dn2v/" + ds_name_2)
    emb_hub = FSEmbedding(f"../../tigra_emb/deep_hub/" + ds_name)
    emb_lid = FSEmbedding(f"../../tigra_emb/deep_lid/" + ds_name)

    #gref1_dn2v = greval(dg, emb_dn2v)
    #gref1_hub = greval(dg, emb_hub)
    #gref1_lid = greval(dg, emb_lid)

    return dg, emb_dn2v, emb_hub, emb_lid   #, gref1_dn2v, gref1_hub, gref1_lid

def diffusion_eval(emb, dg, beta=0.2, seed=0):
    si = SI(verbose=False, beta=beta)
    dp = DiffusionPrediction(emb, dg, si, negative_sampling=True, verbose=False, seed=seed)
    f1, p, r, cl = dp.evaluate()

    dp2 = DiffusionPrediction(emb, dg, si, negative_sampling=False, verbose=False, seed=seed)
    f1_2, p_2, r_2, cl_2 = dp2.evaluate()

    if f1 > f1_2:
        return f1, p, r, cl, True
    else:
        return f1_2, p_2, r_2, cl_2, False


def difeval(dataset, num_runs=1):
    dg, emb_dn2v, emb_hub, emb_lid = get_embeddings(dataset)

    hub_wins, hub_losses, lid_wins, lid_losses = 0, 0, 0, 0

    beta = 0.01
    while beta <= 0.25:
        f1_dn2v, f1_hub, f1_lid = 0, 0, 0
        
        f1s = []
        for i in range(num_runs):
            f1, _, _, _, _ = diffusion_eval(emb_dn2v, dg, beta=beta, seed=i)
            f1s.append(f1)
        f1_dn2v = np.mean(f1s)
        
        f1s = []
        for i in range(num_runs):
            f1, _, _, _, _ = diffusion_eval(emb_hub, dg, beta=beta, seed=i)
            f1s.append(f1)
        f1_hub = np.mean(f1s)
        
        f1s = []
        for i in range(num_runs):
            f1, _, _, _, _ = diffusion_eval(emb_lid, dg, beta=beta, seed=i)
            f1s.append(f1)
        f1_lid = np.mean(f1s)


        print(round(beta, 4), round(f1_dn2v, 4), round(f1_hub, 4), round(f1_lid, 4))
    
        if f1_hub > f1_dn2v:
            hub_wins += 1
        elif f1_hub < f1_dn2v:
            hub_losses += 1

        if f1_lid > f1_dn2v:
            lid_wins += 1
        elif f1_lid < f1_dn2v:
            lid_losses += 1

        beta += 0.01

    print("HUB wins losses", hub_wins, hub_losses)
    print("LID wins losses", lid_wins, lid_losses)


if __name__ == "__main__":
    difeval(datasets[1], num_runs=10)