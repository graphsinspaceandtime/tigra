import sys
import os

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from data.bank import Databank
from tigra.core.loaders import load_event_sequence
from tigra.core.greval import GraphReconstructionEval
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.rndw.tempwalk import DynDeepWalkBF, DynDeepLIDWalkBF
from tigra.embeddings.sota.factory import get_dn2v_factory

db = Databank()
datasets = db.get_keys()
print(datasets)

NUM_RUNS = 5

DATASET = "email-Eu-core-temporal"
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


def run_deeplid(inverse, log_scaling, p, u):
    print("DYNLIDDEEP")
    graph_sequence, dg = get_graph_sequence()
    num_snaps = len(graph_sequence)
    prec, rec, f1 = [0] * num_snaps, [0] * num_snaps, [0] * num_snaps

    print(params[3], params[4])

    for _ in range(NUM_RUNS):
        elid = DynDeepLIDWalkBF(
            dim=params[0],
            walk_length=params[3],
            num_walks=params[4],
            inverse=inverse,
            log_scaling=log_scaling,
            p=p,
            u=u,
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
        NUM_RUNS = 5
        params = dn2v_fact.configs[DATASET]
        print("DATASET", DATASET)
        print("PARAMS", params)
        print()

        print("DEFAULT, no inverse, no log-scaling")
        run_deeplid(False, False, 0.5, 0.25)

        print("log-scaling only")
        run_deeplid(False, True, 0.5, 0.25)

        print("inverse only")
        run_deeplid(True, False, 0.5, 0.25)

        print("inverse + log-scaling")
        run_deeplid(True, True, 0.5, 0.25)

        print()
        print()
        print()
