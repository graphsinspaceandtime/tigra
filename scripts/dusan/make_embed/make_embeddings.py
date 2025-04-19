from tigra.embeddings.sota.de_factory import get_dynae_factory
from tigra.embeddings.embedding import FSEmbedding

dynae_factory = get_dynae_factory("../../../dynae_config.json", "../../../data")

for dataset, config in dynae_factory.dataset_configs.items():
    if config["skip"]:
        print(f"WARNING: Skipping {dataset}")
        continue

    print(f"DATASET: {dataset}")

    embed = dynae_factory.get_embedding(dataset)

    if embed == None:
        print(f"WARNING: Embedding for {dataset} not generated")
        continue

    dynae, _ = embed
    dynae.save_embedding(f"embs/{dataset}")
    print(f"Embedding for {dataset} saved")
