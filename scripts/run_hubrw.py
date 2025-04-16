import sys
import os

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from data.bank import Databank
from tigra.core.loaders import load_event_sequence
from tigra.core.greval import GraphReconstructionEval
from tigra.core.dgraph import DynamicGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.rndw.hubrw import (
        DynN2V_HubDelta,
        DynN2V_HubDeltaNWWL,
        DynN2V_HubDeltaInverseDegree,
        DynN2V_HubDeltaLogDegree,
        DynN2V_HubDeltaMaxDegree,
        DynN2V_HubDeltaMaxLogDegree,
        DynN2V_HubDeltaInverseLogDegree,
        DynN2V_HubDeltaSqrtDegree
    )

from tigra.embeddings.embedding import FSEmbedding
from tigra.embeddings.sota.factory import get_dn2v_factory


db = Databank()
datasets = db.get_keys()
print(datasets)

NUM_RUNS = 10

DATASET = "radoslaw-email"
dn2v_fact = get_dn2v_factory("../DN2V_SVC.conf")
params = dn2v_fact.configs[DATASET]
print("DATASET", DATASET)
print("PARAMS", params)
print()


def eval(emb, graph_sequence):
    eval_results = []
    # print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(graph_sequence)):
        snap = emb.embeddings[i]
        graph = graph_sequence[i]
        gre = GraphReconstructionEval(graph, snap)
        prec = gre.get_avg_precision()
        rec = gre.get_avg_recall()
        f1 = gre.getF1()
        # print(i, prec, rec, f1)
        eval_results.append((prec, rec, f1))

    return eval_results


def get_graph_sequence():
    databank = Databank()
    es, resol = load_event_sequence(DATASET)
    # print("num-events=", es.num_events())
    # print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    # dg.info()

    graph_sequence = dg.get_graph_sequence()
    return graph_sequence, dg


def run_hubdelta():
    print("DynN2V_HubDelta")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDelta(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



def run_hubdeltanwwl():
    print("DynN2V_HubDeltaNWWL")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDeltaNWWL(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



def run_hubdeltainversedegree():
    print("DynN2V_HubDeltaInverseDegree")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDeltaInverseDegree(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



def run_hubdeltalogdegree():
    print("DynN2V_HubDeltaLogDegree")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDeltaLogDegree(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



def run_hubdeltamaxdegree():
    print("DynN2V_HubDeltaMaxDegree")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDeltaMaxDegree(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



def run_hubdeltamaxlogdegree():
    print("DynN2V_HubDeltaMaxLogDegree")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDeltaMaxLogDegree(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



def run_hubdeltainverselogdegree():
    print("DynN2V_HubDeltaInverseLogDegree")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDeltaInverseLogDegree(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



def run_hubdeltasqrtdegree():
    print("DynN2V_HubDeltaSqrtDegree")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        ehub = DynN2V_HubDeltaSqrtDegree(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        ehub.make_embedding(dg)
        eval_res = eval(ehub, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)



if __name__ == "__main__":
    # run_hubdelta()
    # run_hubdeltainversedegree()
    # run_hubdeltainverselogdegree()
    # run_hubdeltalogdegree()
    # run_hubdeltamaxdegree()
    # run_hubdeltamaxlogdegree()
    # run_hubdeltanwwl()
    # run_hubdeltasqrtdegree()


    datasets = [
        "radoslaw-email",
        "fb-forum",
        "fb-messages",
        "ia-contacts_hypertext2009",
        "ia-enron-employees",
        "ia-hospital-ward-proximity-attr",
        "ia-primary-school-proximity-attr",
        "ia-reality-call",
        "CollegeMsg",
        "email-Eu-core-temporal",
    ]

    # for d in datasets:
    #     DATASET = d
    #     NUM_RUNS = 10
    #     params = dn2v_fact.configs[DATASET]
    #     print()
    #     print("DATASET", DATASET)
    #     print(params)
    #     run_hubdeltanwwl()

    d = int(sys.argv[1].strip())

    DATASET = datasets[d]
    NUM_RUNS = 10
    params = dn2v_fact.configs[DATASET]
    print()
    print("DATASET", DATASET)
    print(params)

    run_hubdelta()
    run_hubdeltainversedegree()
    run_hubdeltainverselogdegree()
    run_hubdeltalogdegree()
    run_hubdeltamaxdegree()
    run_hubdeltamaxlogdegree()
    run_hubdeltanwwl()
    run_hubdeltasqrtdegree()