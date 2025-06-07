"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Classes for dynamic graph embedding methods based on random walks
"""

from abc import ABC, abstractmethod

import typing
import random
import bisect
import math

import networkx as nx
import numpy as np

from itertools import chain
from typing import List

from multiprocessing import Pool

from gensim.models import Word2Vec

from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.embedding import Embedding, SingletonEmbedding

from tigra.embeddings.rndw.rndwbase import RandomWalkSampler, UnbiasedRandomWalk

from tigra.evaluation.lideval import NCLIDEstimator



class RWEmbBaseBF(Embedding):
    def __init__(self, dim, delta_nodes=True, num_workers=1):
        """
        dim -- embedding dimension

        delta_nodes -- if false random walks are sampled for all nodes, if true then
            random walks are sampled for a subset of nodes returned by determine_delta_nodes

        num_workers -- the number of parallel processes when constructing embeddings
        """
        super().__init__(dim)
        self.delta_nodes = delta_nodes
        self.num_workers = num_workers
        self.delta_fraction = []

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
            next = None
            if i + 1 < num_graphs:
                next = self.graphs[i + 1]
            self.update_model(self.graphs[i], self.graphs[i - 1], next)
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

    def update_model(self, graph: nx.Graph, prev_graph: nx.Graph, next_graph: nx.Graph):
        if self.delta_nodes:
            nodes = self.determine_delta_nodes(graph, prev_graph, next_graph)
            frac = len(nodes) / float(len(graph.nodes()))
            self.delta_fraction.append(frac)
        else:
            nodes = list(graph.nodes())

        rws = self.create_random_walk_sampler(graph, nodes)
        walks = rws.sample_walks()

        self.model.build_vocab(walks, update=True)
        self.model.train(walks, total_examples=len(walks), epochs=self.model.epochs)

    def get_delta_fraction(self):
        return self.delta_fraction

    @abstractmethod
    def create_random_walk_sampler(self, graph: nx.Graph, nodes) -> RandomWalkSampler:
        pass

    @abstractmethod
    def determine_delta_nodes(self, graph: nx.Graph, prev_graph: nx.Graph, next_graph: nx.Graph) -> List[int]:
        pass



class RWEmbEgoDeltaBF(RWEmbBaseBF):
    """
    Delta nodes = new nodes + nodes whose ego networks changed + future new nodes
    """

    def __init__(self, dim, delta_nodes=True, num_workers=1):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)

    def determine_delta_nodes(self, graph: nx.Graph, prev_graph: nx.Graph, next_graph: nx.Graph) -> List[int]:
        if next_graph != None:
            delta_edges = graph.edges ^ prev_graph.edges ^ next_graph.edges
        else:
            delta_edges = graph.edges ^ prev_graph.edges
        nodes_with_modified_edges = set(chain(*delta_edges))
        delta_nodes = list(graph.nodes & nodes_with_modified_edges)
        return delta_nodes
    

class DynDeepWalkBF(RWEmbEgoDeltaBF):
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


class LIDDeepSampler(RandomWalkSampler):
    def __init__(
        self,
        graph,
        nodes,
        walk_length=32,
        num_walks=10,
        num_workers=1,
        inverse=False,
        log_scaling=False,
        p=0,
        u=0,
    ):
        super().__init__(graph, nodes, num_workers)
        self.graph = graph
        self.nodes = nodes
        self.wl = walk_length
        self.nw = num_walks
        self.num_workers = num_workers
        self.inverse = inverse
        self.log_scaling = log_scaling
        self.p = p  # probability to return back in walk
        self.u = u  # probability to select node uniformly from its neighborhood

        self.lidestimator = NCLIDEstimator(graph)
        self.lidestimator.estimate_lids()
        self.lids = self.lidestimator.get_lid_values()

    def select_next_node(self, start_node, current_node, prev_node, neighbours):

        r = random.random()
        if r < self.p:
            return prev_node

        r = random.random()
        if r < self.u:
            return random.sample(neighbours, 1)[0]

        probabilities = [0.0] * len(neighbours)

        max_lid = 0.0
        for n in neighbours:
            if self.lids[n] > max_lid:
                max_lid = self.lids[n]

        if max_lid == 0:
            return np.random.choice(neighbours, 1)[0]

        for i in range(len(neighbours)):
            n = neighbours[i]
            lid = self.lids[n]
            if self.inverse:
                probabilities[i] = 1 + max_lid - lid
            else:
                probabilities[i] = 1 + lid

        if self.log_scaling:
            for i in range(len(neighbours)):
                probabilities[i] = math.log(1 + probabilities[i])

        total = 0.0
        for p in probabilities:
            total += p

        if total == 0.0:
            return np.random.choice(neighbours, 1)[0]

        probs = [p / total for p in probabilities]

        return np.random.choice(neighbours, 1, p=probs)[0]

    def num_walks(self, node) -> int:
        return self.nw

    def walk_length(self, node) -> int:
        return self.wl


class DynDeepLIDWalkBF(RWEmbEgoDeltaBF):
    def __init__(
        self,
        dim,
        delta_nodes=True,
        num_walks=10,
        walk_length=32,
        num_workers=1,
        inverse=False,
        log_scaling=False,
        p=0,
        u=0,
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.inverse = inverse
        self.log_scaling = log_scaling
        self.p = p
        self.u = u

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return LIDDeepSampler(
            graph,
            nodes,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
            inverse=self.inverse,
            log_scaling=self.log_scaling,
            p=self.p,
            u=self.u,
        )
