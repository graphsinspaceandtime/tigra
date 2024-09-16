from typing import List

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
