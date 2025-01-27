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
    es, resol = load_event_sequence("radoslaw-email")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    from tigra.embeddings.rndw.lidbasedrw import LIDAwareEmbedding

    start = time.time_ns()
    lrw = LIDAwareEmbedding(50)
    lrw.make_embedding(dg)

    

    graph_sequence = dg.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb = lrw.embeddings[i]
        graph = graph_sequence[i]
        
        gre = GraphReconstructionEval(graph, emb)

        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())

    end = time.time_ns()
    print(end - start)
