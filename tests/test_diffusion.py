import sys

sys.path.append("..")
sys.dont_write_bytecode = True

if __name__ == "__main__":
    import numpy as np
    from tigra.core.dgraph import DynamicGraph
    from tigra.core.loaders import load_event_sequence
    from tigra.embeddings.embedding import FSEmbedding
    from tigra.core.greval import GraphReconstructionEval

    es, resol = load_event_sequence("radoslaw-email")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    #emb = FSEmbedding(f"../../tigra_emb/dn2v/radoslaw_email")
    #emb = FSEmbedding(f"../../tigra_emb/deep_hub/radoslaw-email")
    emb = FSEmbedding(f"../../tigra_emb/deep_lid/radoslaw-email")
    print("#embeddings", emb.num_embeddings())

    graph_sequence = dg.get_graph_sequence()
    f1_list = []
    for i in range(len(graph_sequence)):
        emb_i = emb.embeddings[i]
        graph = graph_sequence[i]
        gre = GraphReconstructionEval(graph, emb_i)
        f1 = gre.getF1()
        f1_list.append(f1)
    

    from tigra.evaluation.diffusion import DiffusionPrediction
    from tigra.evaluation.diffusion import SI

    si = SI(verbose=False)
    dp = DiffusionPrediction(emb, dg, si, negative_sampling=True, verbose=True)
    dp.init_evaluation()
    f1, p, r, cl = dp.evaluate()

    print("GRE-eval", round(np.mean(f1_list), 4), round(f1_list[-1], 4))
    print("DIF-eval", round(f1, 4), round(p, 4), round(r, 4), " -- ", cl)
