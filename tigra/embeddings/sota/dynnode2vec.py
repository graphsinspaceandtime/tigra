"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Wrapper for dynnode2vec
"""

from typing import List
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicGraph

from dynnode2vec.dynnode2vec import DynNode2Vec


class Dynnode2vec(Embedding):
    def __init__(
        self,
        dim,
        p=1.0,
        q=1.0,
        walk_length=10,
        n_walks_per_node=10,
        parallel_processes=4,
    ):
        super().__init__(dim)
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node
        self.parallel_processes = parallel_processes

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicGraph):
            raise Exception(
                "{} works with DynamicGraph objects.".format(type(self).__name__)
            )

        if data.is_directed():
            raise Exception(
                "{} works with undirected graphs.".format(type(self).__name__)
            )

        dynnode2vec = DynNode2Vec(
            p=self.p,
            q=self.q,
            walk_length=self.walk_length,
            n_walks_per_node=self.n_walks_per_node,
            embedding_size=self.dim,
            parallel_processes=self.parallel_processes,
        )

        graph_sequence = data.get_nx_graph_sequence()
        embs = dynnode2vec.compute_embeddings(graph_sequence)

        embeddings = []
        for emb_num in range(len(embs)):
            emb = embs[emb_num]
            vectors = emb[0]
            mapping = emb[1]
            num_vectors = len(vectors)
            singleton = SingletonEmbedding()

            orgG = graph_sequence[emb_num]

            for i in range(num_vectors):
                id = mapping[i]
                if id in orgG.nodes():
                    vect = vectors[i].tolist()
                    singleton.add_embedding_vector(id, vect)

            embeddings.append(singleton)

        return embeddings
