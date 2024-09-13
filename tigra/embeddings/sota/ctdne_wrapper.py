"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Wrapper for CTDNE
"""

import networkx as nx
from typing import List

from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicAggregateGraph
from tigra.embeddings.sota.ctdnesrc.ctdne import CTDNE


class Ctdne(Embedding):
    def __init__(
        self,
        dim,
        p=1.0,
        q=1.0,
        walk_length=10,
        num_walks=10,
        workers=4,
        window=10,
        min_count=1,
        batch_words=4,
    ):
        super().__init__(dim)
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers

        # skip-gram params
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicAggregateGraph):
            raise Exception(
                "{} works with DynamicAggregateGraph objects.".format(
                    type(self).__name__
                )
            )

        graph = data.get_aggregate_graph().G
        if not isinstance(graph, nx.Graph):
            raise Exception(
                "{} works with undirected graphs.".format(type(self).__name__)
            )

        edge2time = dict()
        for e in graph.edges():
            edge2time[e] = graph.edges[e]["edge_data"][0]

        nx.set_edge_attributes(graph, edge2time, "time")
        CTDNE_model = CTDNE(
            graph,
            p=self.p,  # type: ignore
            q=self.q,  # type: ignore
            dimensions=self.dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            quiet=True,
        )

        model = CTDNE_model.fit(
            window=self.window, min_count=self.min_count, batch_words=self.batch_words
        )

        singleton = SingletonEmbedding()

        for node in graph.nodes():
            key = str(node)
            vector = model.wv[key].tolist()
            singleton.add_embedding_vector(node, vector)

        return [singleton]
