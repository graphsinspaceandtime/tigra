from tigra.embeddings.sota.de_factory import get_dynaernn_factory
from tigra.embeddings.embedding import FSEmbedding

factory = get_dynaernn_factory("../../../dynaernn_config.json", "../../../data")

for dataset, config in factory.dataset_configs.items():
    if config["skip"]:
        print(f"WARNING: Skipping {dataset}")
        continue

    print(f"DATASET: {dataset}")

    embed = factory.get_embedding(dataset)

    if embed == None:
        print(f"WARNING: Embedding for {dataset} not generated")
        continue

    dynae, _ = embed
    dynae.save_embedding(f"embs_aernn/{dataset}")
    print(f"Embedding for {dataset} saved")
