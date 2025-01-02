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
    es, resol = load_event_sequence("fb-forum")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    from tigra.embeddings.rndw.rndwbase import DynDeepWalk, DynN2V

    start = time.time_ns()
    ddw = DynDeepWalk(50)
    ddw.make_embedding(dg)

    dn2v = DynN2V(200, 1, 1)
    dn2v.make_embedding(dg)

    graph_sequence = dg.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb = ddw.embeddings[i]
        emb_2 = dn2v.embeddings[i]
        graph = graph_sequence[i]

        gre = GraphReconstructionEval(graph, emb)
        gre_2 = GraphReconstructionEval(graph, emb_2)

        print("Precision", gre.get_avg_precision(), gre_2.get_avg_precision())
        print("Recall", gre.get_avg_recall(), gre_2.get_avg_recall())
        print("F1", gre.getF1(), gre_2.getF1())

    end = time.time_ns()
    print(end - start)

    # from tigra.embeddings.rndw.rndwbase import PQRandomWalk
    # pqrw = PQRandomWalk(graph_sequence[0].G, list(graph_sequence[0].G.nodes()), 0.5, 4)
    # print(pqrw.transition_weight)
