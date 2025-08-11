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

from scipy.stats import ttest_rel

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

def diffusion_eval(emb, dg, beta=0.2, seed=0, model_type="ALL"):
    si = SI(verbose=False, beta=beta)
    dp = DiffusionPrediction(emb, dg, si, negative_sampling=True, verbose=False, seed=seed)
    dp.init_evaluation()
    f1, p, r, cl = dp.evaluate(model_type=model_type)

    dp2 = DiffusionPrediction(emb, dg, si, negative_sampling=False, verbose=False, seed=seed)
    dp2.init_evaluation()
    f1_2, p_2, r_2, cl_2 = dp2.evaluate(model_type=model_type)

    if f1 > f1_2:
        return f1, p, r, cl, True
    else:
        return f1_2, p_2, r_2, cl_2, False


def difeval_baseline(dataset, num_runs=1, model_type="ALL"):
    dg, emb, _, _ = get_embeddings(dataset)

    f2 = model_type + "-sam"
    f3 = model_type + "-nosam"
    f4 = model_type

    print("BETA," + f2 + "," + f3 + "," + f4)
    beta = 0.01
    while beta <= 0.26:
        f1s_ns, f1s_full, f1s = [], [], []
        for i in range(num_runs):
            si = SI(verbose=False, beta=beta)
            dp = DiffusionPrediction(emb, dg, si, negative_sampling=True, verbose=False, seed=i)
            dp.init_evaluation()
            f1, _, _, _ = dp.evaluate(model_type=model_type)
            f1s_ns.append(f1)

            si2 = SI(verbose=False, beta=beta)
            dp2 = DiffusionPrediction(emb, dg, si2, negative_sampling=False, verbose=False, seed=i)
            dp2.init_evaluation()
            f1_2, _, _, _ = dp2.evaluate(model_type=model_type)
            f1s_full.append(f1_2)

            f1_fin = f1 if f1 > f1_2 else f1_2
            f1s.append(f1_fin)

        f1_ns = np.mean(f1s_ns)
        f1_full = np.mean(f1s_full)
        f1 = np.mean(f1s)

        s = str(round(beta, 4)) + "," + str(round(f1_ns, 4)) + "," + str(round(f1_full, 4)) + "," + str(round(f1, 4))
        print(s)

        beta += 0.01

def diffeval_baselines(dataset, num_runs=1):
    print("RF")
    difeval_baseline(dataset, num_runs=num_runs, model_type="RF")

    print()
    print("KNN")
    difeval_baseline(dataset, num_runs=num_runs, model_type="KNN")

    print()
    print("NB")
    difeval_baseline(dataset, num_runs=num_runs, model_type="NB")

    print()
    print("SVM")
    difeval_baseline(dataset, num_runs=num_runs, model_type="SVM")

    print()
    print("LR")
    difeval_baseline(dataset, num_runs=num_runs, model_type="LR")


def difeval(dataset, num_runs=1, model_type="ALL"):
    dg, emb_dn2v, emb_hub, emb_lid = get_embeddings(dataset)

    hub_wins, hub_losses, lid_wins, lid_losses = 0, 0, 0, 0
    ss_hub_wins, ss_hub_losses, ss_lid_wins, ss_lid_losses = 0, 0, 0, 0

    print(dataset)
    print("BETA,F1-DN2V,F1-DeepHUB,F1-DeepLID,SS-HUB,SS-LID,p-HUB,p-LID")

    DN2V_f1, HUB_f1, LID_f1 = [], [], []

    beta = 0.01
    while beta <= 0.26:
        f1_dn2v, f1_hub, f1_lid = 0, 0, 0
        
        f1s_DN2V = []
        for i in range(num_runs):
            f1, _, _, _, _ = diffusion_eval(emb_dn2v, dg, beta=beta, seed=i, model_type=model_type)
            f1s_DN2V.append(f1)
            DN2V_f1.append(f1)
        f1_dn2v = np.mean(f1s_DN2V)
        
        f1s_HUB = []
        for i in range(num_runs):
            f1, _, _, _, _ = diffusion_eval(emb_hub, dg, beta=beta, seed=i, model_type=model_type)
            f1s_HUB.append(f1)
            HUB_f1.append(f1)
        f1_hub = np.mean(f1s_HUB)
        
        f1s_LID = []
        for i in range(num_runs):
            f1, _, _, _, _ = diffusion_eval(emb_lid, dg, beta=beta, seed=i, model_type=model_type)
            f1s_LID.append(f1)
            LID_f1.append(f1)
        f1_lid = np.mean(f1s_LID)

        ss_res_hub = ttest_rel(f1s_DN2V, f1s_HUB)
        hub_ssdif = True if ss_res_hub.pvalue < 0.05 else False 
    
        ss_res_lid = ttest_rel(f1s_DN2V, f1s_LID)
        lid_ssdif = True if ss_res_lid.pvalue < 0.05 else False

        s = str(round(beta, 4)) + "," 
        s += str(round(f1_dn2v, 4)) + "," 
        s += str(round(f1_hub, 4)) + ","
        s += str(round(f1_lid, 4)) + ","  
        s += str(hub_ssdif) + "," 
        s += str(lid_ssdif) + ","
        s += str(round(ss_res_hub.pvalue, 4)) + ","
        s += str(round(ss_res_lid.pvalue, 4))

        print(s)
    
        if f1_hub > f1_dn2v:
            hub_wins += 1
            if hub_ssdif:
                ss_hub_wins += 1
        elif f1_hub < f1_dn2v:
            hub_losses += 1
            if hub_ssdif:
                ss_hub_losses += 1

        if f1_lid > f1_dn2v:
            lid_wins += 1
            if lid_ssdif:
                ss_lid_wins += 1
        elif f1_lid < f1_dn2v:
            lid_losses += 1
            if lid_ssdif:
                ss_lid_losses += 1

        beta += 0.01

    print("HUB wins losses", hub_wins, hub_losses, ss_hub_wins, ss_hub_losses)
    print("LID wins losses", lid_wins, lid_losses, ss_lid_wins, ss_lid_losses)

    f1_dn2v = np.mean(DN2V_f1)
    f1_dn2v_std = np.std(DN2V_f1)

    f1_hub = np.mean(HUB_f1)
    f1_hub_std = np.std(HUB_f1)

    f1_lid = np.mean(LID_f1)
    f1_lid_std = np.std(LID_f1)

    ss_res_hub = ttest_rel(DN2V_f1, HUB_f1)
    hub_ssdif = True if ss_res_hub.pvalue < 0.05 else False 
    
    ss_res_lid = ttest_rel(DN2V_f1, LID_f1)
    lid_ssdif = True if ss_res_lid.pvalue < 0.05 else False

    s = str(round(f1_dn2v, 4)) + "," + str(round(f1_dn2v_std, 4)) + ","
    s += str(round(f1_hub, 4)) + "," + str(round(f1_hub_std, 4)) + ","
    s += str(round(f1_lid, 4)) + "," + str(round(f1_lid_std, 4)) + ","
    s += str(hub_ssdif) + "," 
    s += str(lid_ssdif) + ","
    s += str(round(ss_res_hub.pvalue, 4)) + ","
    s += str(round(ss_res_lid.pvalue, 4))
    print(s) 


def difeval_allc(dataset, num_runs=1):
    print("RF")
    difeval(dataset, num_runs=num_runs, model_type="RF")

    print()
    print("KNN")
    difeval(dataset, num_runs=num_runs, model_type="KNN")

    print()
    print("NB")
    difeval(dataset, num_runs=num_runs, model_type="NB")

    print()
    print("SVM")
    difeval(dataset, num_runs=num_runs, model_type="SVM")

    print()
    print("LR")
    difeval(dataset, num_runs=num_runs, model_type="LR")

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

if __name__ == "__main__":
    index = int(sys.argv[1])
    print(datasets[index])
    print()

    #difeval(datasets[0], num_runs=20, model_type="RF")
    #difeval_baseline(datasets[0], num_runs=5, model_type="RF")
    #diffeval_baselines(datasets[index], num_runs=25)

    difeval_allc(datasets[index], num_runs=20)