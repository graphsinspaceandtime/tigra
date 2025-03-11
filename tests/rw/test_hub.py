import sys
import os

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

if __name__ == "__main__":
    from tigra.core.dgraph import DynamicGraph
    from data.bank import Databank
    from tigra.core.greval import GraphReconstructionEval
    import time
    from tigra.core.loaders import load_event_sequence

    databank = Databank()
    es, resol = load_event_sequence("fb-messages")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    from tigra.embeddings.rndw.hubrw import (
        HubRandomWalk,
        HubChangedRandomWalk,
        DynN2V_HubDelta,
        DynN2V_HubDeltaNWWL,
        DynN2V_HubDeltaInverseDegree,
        DynN2V_HubDeltaLogDegree,
        HubChangedRandomWalkReverse,
        DynN2V_HubDeltaMaxDegree,
        DynN2V_HubDeltaMaxLogDegree,
        DynN2V_HubDeltaInverseLogDegree,
    )

    start = time.time_ns()

    dn2v = DynN2V_HubDeltaMaxDegree(100, 0.5, 1)
    dn2v.make_embedding(dg)

    graph_sequence = dg.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb_2 = dn2v.embeddings[i]
        graph = graph_sequence[i]

        gre_2 = GraphReconstructionEval(graph, emb_2)

        print("Precision", gre_2.get_avg_precision())
        print("Recall", gre_2.get_avg_recall())
        print("F1", gre_2.getF1())

    end = time.time_ns()
    print(end - start)
