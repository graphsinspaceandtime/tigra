"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Base classes for dynamic graph embedding methods based on random walks and reinforcement learning

Author: atomic@dmi.uns.ac.rs
"""

import networkx as nx
import numpy as np
import random
from tigra.embeddings.rndw.rndwbase import RWEmbEgoDelta, RandomWalkSampler


class BanditSampler(RandomWalkSampler):
    def __init__(
        self,
        graph: nx.Graph,
        nodes,
        epsilon=0.2,
        alpha=0.1,
        reward_bias=True,
        num_walks=10,
        walk_length=40,
        num_workers=1,
    ):
        super().__init__(graph, nodes, num_workers=num_workers)
        self.epsilon = epsilon
        self.alpha = alpha
        self.reward_bias = reward_bias
        self.nw = num_walks
        self.wl = walk_length
        self.q_values = [0.5, 0.5]
        self._neighbor_cache = {}

    def _get_neighbors_set(self, node):
        if node not in self._neighbor_cache:
            self._neighbor_cache[node] = set(self.G.neighbors(node))
        return self._neighbor_cache[node]

    def _select_structural(self, current_node, neighbours):
        if len(neighbours) == 1:
            return neighbours[0]
        set_curr = self._get_neighbors_set(current_node)
        weights = [len(set_curr & self._get_neighbors_set(n)) + 0.1 for n in neighbours]
        total = sum(weights)
        return np.random.choice(neighbours, p=[w / total for w in weights])

    def _select_inverse(self, neighbours):
        if len(neighbours) == 1:
            return neighbours[0]
        degrees = [self.G.degree[n] for n in neighbours]
        weights = [1.0 / (d + 1.0) for d in degrees]
        total = sum(weights)
        return np.random.choice(neighbours, p=[w / total for w in weights])

    def rnd_walks_from_node(self, start_node, walk_length):
        walk = [start_node]
        visited = {start_node}
        curr = start_node
        while len(walk) < walk_length:
            neis = list(self.G.neighbors(curr))
            if not neis:
                break

            if random.random() < self.epsilon:
                action = random.choice([0, 1])
            else:
                action = np.argmax(self.q_values)

            next_node = (
                self._select_structural(curr, neis)
                if action == 0
                else self._select_inverse(neis)
            )

            if next_node not in visited:
                reward = (
                    (1.0 + (5.0 / (self.G.degree[next_node] + 1.0)))
                    if self.reward_bias
                    else 1.0
                )
                visited.add(next_node)
            else:
                reward = -0.1

            self.q_values[action] += self.alpha * (reward - self.q_values[action])
            walk.append(next_node)
            curr = next_node
        return walk

    def select_next_node(self, s, c, p, n):
        pass

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class DynRLWalk(RWEmbEgoDelta):
    def __init__(
        self,
        dim,
        epsilon=0.1,
        alpha=0.1,
        reward_bias=False,
        num_walks=10,
        walk_length=40,
        num_workers=1,
        **kwargs,
    ):
        super().__init__(dim, delta_nodes=True, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.alpha = alpha
        self.reward_bias = reward_bias

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return BanditSampler(
            graph,
            nodes,
            epsilon=self.epsilon,
            alpha=self.alpha,
            reward_bias=self.reward_bias,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )
