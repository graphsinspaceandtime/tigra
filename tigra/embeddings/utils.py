from typing import Union, List, Tuple, Generator
from tigra.embeddings.embedding import SingletonEmbedding


def n_nodes_to_ae_layers(n_nodes: int, embedding_size: int) -> List[int]:
    top_layer = 2
    while n_nodes > top_layer:
        top_layer <<= 1
    top_layer >>= 1
    layers = []
    while top_layer > embedding_size:
        layers.append(top_layer)
        top_layer >>= 1
    return layers


def iterate_valid_embeddings(
    embeddings: List[Union[SingletonEmbedding, None]]
) -> Generator[Tuple[int, SingletonEmbedding], None, None]:
    for i, embedding in enumerate(embeddings):
        if embedding != None:
            yield (i, embedding)
