"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Base classes for dynamic graph embedding methods based on random walks
"""

from abc import ABC, abstractmethod
import typing
import random
from itertools import chain
import networkx as nx
from typing import List
from multiprocessing import Pool

from gensim.models import Word2Vec

from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.embedding import Embedding, SingletonEmbedding


class RandomWalkSampler:
    """
    Base class for random walk samplers
    """

    def __init__(self, graph: nx.Graph, nodes, num_workers=1):
        self.G = graph
        self.nodes = nodes
        self.num_workers = num_workers

    @abstractmethod
    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        pass

    @abstractmethod
    def num_walks(self, node) -> int:
        pass

    @abstractmethod
    def walk_length(self, node) -> int:
        pass

    def rnd_walks_from_node(self, start_node, walk_length):
        walk = [start_node]
        prev_node = None

        while len(walk) < walk_length:
            current_node = walk[-1]
            current_neis = [edge[1] for edge in self.G.edges(current_node)]
            if len(current_neis) == 0:
                break

            next_node = self.select_next_node(
                start_node, current_node, prev_node, current_neis
            )
            if next_node == None:
                break

            walk.append(next_node)
            prev_node = current_node

        return walk

    def sample_rnd_walks_from_node(self, node):
        walks = []
        num_walks = self.num_walks(node)
        walk_length = self.walk_length(node)
        for _ in range(num_walks):
            walks.append(
                self.rnd_walks_from_node(walk_length=walk_length, start_node=node)
            )

        return walks

    def sample_walks(self):
        walks = []
        if self.num_workers == 1:
            for node in self.nodes:
                w = self.sample_rnd_walks_from_node(node)
                walks.extend(w)
        else:
            pool = Pool(self.num_workers)
            wlk = pool.map(func=self.sample_rnd_walks_from_node, iterable=self.nodes)
            walks = [w for node_walks in wlk for w in node_walks]

        random.shuffle(walks)

        return walks


class RWEmbBase(Embedding):
    def __init__(self, dim, delta_nodes=True, num_workers=1):
        """
        dim -- embedding dimension

        delta_nodes -- if false random walks are sampled for all nodes, if true then
            random walks are sampled for nodes whose neighboorhods changed w.r.t
            the previous evolutionary snapshot

        num_workers -- the number of parallel processes when constructing embeddings
        """
        super().__init__(dim)
        self.delta_nodes = delta_nodes
        self.num_workers = num_workers

    def construct(self, data) -> List[typing.Union[SingletonEmbedding, None]]:
        if not isinstance(data, DynamicGraph):
            raise Exception(
                "{} works with DynamicGraph objects.".format(type(self).__name__)
            )

        if data.is_directed():
            raise Exception(
                "{} works with undirected graphs.".format(type(self).__name__)
            )

        self.graphs = data.get_nx_graph_sequence()
        num_graphs = len(self.graphs)
        if num_graphs == 0:
            raise Exception("{} -- empty graph sequence.".format(type(self).__name__))

        embeddings = []

        self.initialize_model(self.graphs[0])
        embeddings.append(self.form_embedding(self.graphs[0], self.model))
        for i in range(1, num_graphs):
            self.update_model(self.graphs[i], self.graphs[i - 1])
            embeddings.append(self.form_embedding(self.graphs[i], self.model))

        return embeddings

    def form_embedding(self, graph, model):
        emb = SingletonEmbedding()
        for node in graph.nodes():
            vector = model.wv[node].tolist()
            emb.add_embedding_vector(node, vector)
        return emb

    def initialize_model(self, first_graph: nx.Graph):
        rws = self.create_random_walk_sampler(first_graph, list(first_graph.nodes()))
        walks = rws.sample_walks()
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.dim,
            window=5,
            min_count=0,
            sg=1,
            workers=self.num_workers,
        )

    def update_model(self, graph: nx.Graph, prev_graph: nx.Graph):
        if self.delta_nodes:
            nodes = self.get_delta_nodes(graph, prev_graph)
        else:
            nodes = list(graph.nodes())

        rws = self.create_random_walk_sampler(graph, nodes)
        walks = rws.sample_walks()

        self.model.build_vocab(walks, update=True)
        self.model.train(walks, total_examples=len(walks), epochs=self.model.epochs)

    def get_delta_nodes(self, graph: nx.Graph, prev_graph: nx.Graph):
        delta_edges = graph.edges ^ prev_graph.edges
        nodes_with_modified_edges = set(chain(*delta_edges))
        delta_nodes = list(graph.nodes & nodes_with_modified_edges)
        return delta_nodes

    @abstractmethod
    def create_random_walk_sampler(self, graph: nx.Graph, nodes) -> RandomWalkSampler:
        pass


class UnbiasedRandomWalk(RandomWalkSampler):
    def __init__(
        self, graph: nx.Graph, nodes, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes)
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers

    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class DynDeepWalk(RWEmbBase):
    def __init__(
        self, dim, delta_nodes=True, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return UnbiasedRandomWalk(
            graph,
            nodes,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )
