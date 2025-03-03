import sys

sys.path.append("..")
sys.dont_write_bytecode = True

if __name__ == "__main__":
    from data.bank import Databank
    from tigra.core.dgraph import DynamicGraph
    from tigra.core.loaders import load_event_sequence
    from tigra.core.greval import GraphReconstructionEval
    from tigra.embeddings.rndw.rndwbase import DynN2V

    databank = Databank()
    es, resol = load_event_sequence("radoslaw-email")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    dn2v = DynN2V(25, 0.5, 4)
    dn2v.make_embedding(dg)
    dn2v.save_embedding("radoslaw-dynn2v-25")

    graph_sequence = dg.get_graph_sequence()

    for i in range(len(graph_sequence)):
        print("SNAPSHOT", i)
        emb = dn2v.embeddings[i]
        graph = graph_sequence[i]

        gre = GraphReconstructionEval(graph, emb)

        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())
