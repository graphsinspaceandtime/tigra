import sys

sys.path.append("..")
sys.dont_write_bytecode = True

if __name__ == "__main__":
    from tigra.core.dgraph import DynamicGraph
    from tigra.core.loaders import load_event_sequence
    from tigra.embeddings.embedding import FSEmbedding

    es, resol = load_event_sequence("radoslaw-email")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    emb = FSEmbedding(f"../../tigra_emb/dn2v/radoslaw_email")
    # emb = FSEmbedding(f"../../tigra_emb/deep_hub/radoslaw-email")
    # emb = FSEmbedding(f"../../tigra_emb/deep_lid/radoslaw-email")
    print("#embeddings", emb.num_embeddings())

    print("\n\nLINK-PREDICTION")

    from tigra.evaluation.linkpred import (
        SupervisedLinkPrediction,
        ConcatinationCombiner,
        Predicter,
    )
    from tigra.evaluation.clbase.skl import RF_SKLC

    spl = SupervisedLinkPrediction(ConcatinationCombiner(), RF_SKLC())
    p = Predicter(spl, emb, dg)
    p.predict(print_results=True)

    print("\n\nDIFFUSION-PREDICTION")

    from tigra.evaluation.diffusion import DiffusionPrediction
    from tigra.evaluation.diffusion import SI

    si = SI(verbose=False)
    dp = DiffusionPrediction(emb, dg, si, negative_sampling=True, verbose=True)
    dp.init_evaluation()
    f1, p, r, cl = dp.evaluate()

    print("DIF-eval", round(f1, 4), round(p, 4), round(r, 4), " -- ", cl)

    print("\n\nCLASSIFICATION")

    # dummy random labeling
    from random import random

    for e in emb:
        for node in e.embedding:
            e.label[node] = "A" if random() <= 0.5 else "B"

    from tigra.evaluation.classification_eval import eval

    cl_res = eval(emb, RF_SKLC())
    print(cl_res)

    cl_res = eval(emb, RF_SKLC(), tt_split=False)
    print(cl_res)
