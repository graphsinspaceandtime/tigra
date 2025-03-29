import sys

sys.path.append("..")
sys.dont_write_bytecode = True

if __name__ == "__main__":
    from data.bank import Databank
    from tigra.core.dgraph import DynamicGraph
    from tigra.core.loaders import load_event_sequence
    from tigra.embeddings.embedding import FSEmbedding

    databank = Databank()
    es, resol = load_event_sequence("radoslaw-email")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    emb = FSEmbedding(f"../../emb/dn2v_radoslaw-email")
    print("#embeddings", emb.num_embeddings())

    from tigra.evaluation.diffusion import DiffusionPrediction
    from tigra.evaluation.diffusion import SI

    si = SI()
    dp = DiffusionPrediction(emb, dg, si, negative_sampling=True, verbose=True)
    dp.evaluate()
