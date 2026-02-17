import sys

sys.path.append("..")
sys.dont_write_bytecode = True

if __name__ == "__main__":
    from data.bank import Databank
    from tigra.core.loaders import load_event_sequence

    databank = Databank()
    es, resol = load_event_sequence("radoslaw-email")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    from tigra.core.dgraph import TemporalAggregateGraph
    tag = TemporalAggregateGraph("radoslaw-email-tag", es)
    print(tag.G)

    """
    from tigra.embeddings.rndw.trndwbase import UnbiasedTempRandomWalk

    utrw = UnbiasedTempRandomWalk(tag)
    walks = utrw.sample_walks()

    print("Total walks", len(walks))
    for w in walks:
        print(w)
    """

    from tigra.embeddings.rndw.trndwbase import TemporalDeepWalk
    from tigra.core.greval import GraphReconstructionEval
    
    DIMS = [10, 25, 50, 100, 200, 500]
    for d in DIMS:
        tdw = TemporalDeepWalk(d)
        tdw.make_embedding(tag)
    
        print("\nGRAPH RECONSTRUCTION")
        print(d)
        gre = GraphReconstructionEval(tag, tdw.embeddings[0])
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())