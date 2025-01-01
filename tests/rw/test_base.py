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
    # es = databank.get_data("radoslaw-email")
    es, resol = load_event_sequence("ia-reality-call")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    from tigra.embeddings.rndw.rndwbase import DynDeepWalk

    start = time.time_ns()
    ddw = DynDeepWalk(10)
    ddw.make_embedding(dg)

    graph_sequence = dg.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb = ddw.embeddings[i]
        graph = graph_sequence[i]
        gre = GraphReconstructionEval(graph, emb)
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())

    end = time.time_ns()
    print(end - start)
