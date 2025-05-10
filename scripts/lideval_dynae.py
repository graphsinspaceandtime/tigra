from tigra.embeddings.sota.de_factory import get_dynae_factory

from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.embedding import FSEmbedding

from tigra.evaluation.lideval import LIDEval

import sys

method = sys.argv[1].strip()

dynae_factory = get_dynae_factory(f"../dyn{method}_config.json")

for dataset, config in dynae_factory.dataset_configs.items():
    if config["skip"]:
        continue

    print("DATASET: " + dataset)

    emb = FSEmbedding(f"dusan/make_embed/embs_{method}/{dataset}")

    Data = get_dataset_by_id(dataset)
    data = Data()

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        event_sequence=data.get_event_sequence(shrink=True),
        time_resolution=data.resolution,
        directed=False,
        cumulative=data.cumulative,
    )

    lideval = LIDEval(emb, dg)
    lideval.eval()

