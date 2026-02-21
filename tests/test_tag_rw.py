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

    """
    from tigra.embeddings.rndw.trndwbase import DeepTemporalWalk
    from tigra.embeddings.rndw.trndw import DeepLIDTemporalWalk, DeepHubTemporalWalk
    from tigra.core.greval import GraphReconstructionEval
    
    DIMS = [10, 25, 50, 100, 200, 500]
    for d in DIMS:
        dtw = DeepTemporalWalk(d)
        dtw.make_embedding(tag)
        print("\nGRAPH RECONSTRUCTION [deep-temporal-walk]")
        print(d)
        gre = GraphReconstructionEval(tag, dtw.embeddings[0])
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())

        dltw = DeepLIDTemporalWalk(d)
        dltw.make_embedding(tag)
        print("\nGRAPH RECONSTRUCTION [deep-lid-temporal-walk]")
        print(d)
        gre = GraphReconstructionEval(tag, dltw.embeddings[0])
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())

        dhtw = DeepHubTemporalWalk(d)
        dhtw.make_embedding(tag)
        print("\nGRAPH RECONSTRUCTION [deep-hub-temporal-walk]")
        print(d)
        gre = GraphReconstructionEval(tag, dhtw.embeddings[0])
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())
    """

    from tigra.embeddings.hypertuner import DeepLIDTemporalWalk_Tuner
    tuner = DeepLIDTemporalWalk_Tuner(tag, tag)
    tuner.tune()
    tuner.print_report()