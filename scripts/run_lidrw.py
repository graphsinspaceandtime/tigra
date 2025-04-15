import sys
import os

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from data.bank import Databank
from tigra.core.loaders import load_event_sequence
from tigra.core.greval import GraphReconstructionEval
from tigra.core.dgraph import DynamicGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.rndw.lidrw import DynN2V_LIDRW, DynN2V_NCDelta, DynN2V_PQLID
from tigra.embeddings.rndw.lidrw import DynN2V_WLLID, DynN2V_NWLID, DynN2V_HLIDDelta
from tigra.embeddings.rndw.rndwbase import DynN2V, DynDeepWalk
from tigra.embeddings.embedding import FSEmbedding
from tigra.embeddings.sota.factory import get_dn2v_factory
from tigra.evaluation.lideval import LIDEval

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


def run_baseline():
    print("BASELINE DYNNODE2VEC")
    graph_sequence, _ = get_graph_sequence()
    emb = FSEmbedding(f"../../emb/dn2v_{DATASET}")
    eval(emb, graph_sequence)


def run_lidrw():
    print("DYNN2V LIDRW")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynN2V_LIDRW(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)


def run_lidncdelta():
    print("DYNN2V NC DELTA")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynN2V_NCDelta(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)


def run_WLLID():
    print("WLLID-DynN2V")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynN2V_WLLID(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)


def run_NWLID(log_scaling=False):
    print("NWLID-DynN2V")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynN2V_NWLID(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
            log_scaling=log_scaling,
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)


def run_PQLID():
    print("PQLID-DynN2V")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynN2V_PQLID(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

        lideval = LIDEval(elid, dg)
        lideval.eval()

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)


def run_HLID_delta():
    print("HLIDDelta")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynN2V_HLIDDelta(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)


def run_baseline2():
    print("DYNN2V BASELINE")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynN2V(
            dim=params[0],
            p=params[1],
            q=params[2],
            walk_length=params[3],
            num_walks=params[4],
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
        for i in range(0, len(eval_res)):
            prec[i] += eval_res[i][0]
            rec[i] += eval_res[i][1]
            f1[i] += eval_res[i][2]

    print()
    print("--> AVERAGED")
    print("SNAPSHOT, PRECISION, RECALL, F1")
    for i in range(len(f1)):
        print(i, prec[i] / NUM_RUNS, rec[i] / NUM_RUNS, f1[i] / NUM_RUNS)


def run_baseline3():
    print("DEEP-WALK BASELINE")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    for _ in range(NUM_RUNS):
        elid = DynDeepWalk(
            dim=params[0],
            walk_length=params[3],
            num_walks=params[4],
        )
        elid.make_embedding(dg)
        eval_res = eval(elid, graph_sequence)
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
    # run_NWLID(log_scaling=True)
    # run_WLLID()
    # run_baseline2()
    # run_baseline2()
    # run_lidrw()
    # run_PQLID()
    # run_baseline3()
    # run_lidncdelta()

    # run_HLID_delta()

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

    for d in datasets:
        DATASET = d
        NUM_RUNS = 10
        params = dn2v_fact.configs[DATASET]
        print()
        print("DATASET", DATASET)
        print(params)
        run_HLID_delta()
