from tigra.embeddings.sota.de_factory import get_dynrnn_factory
from tigra.embeddings.embedding import FSEmbedding

dynrnn_factory = get_dynrnn_factory("../../../dynrnn_config.json", "../../../data")

for dataset, config in dynrnn_factory.dataset_configs.items():
    if config["skip"]:
        print(f"WARNING: Skipping {dataset}")
        continue

    print(f"DATASET: {dataset}")

    embed = dynrnn_factory.get_embedding(dataset)

    if embed == None:
        print(f"WARNING: Embedding for {dataset} not generated")
        continue

    dynae, _ = embed
    dynae.save_embedding(f"embs_rnn/{dataset}")
    print(f"Embedding for {dataset} saved")
