"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Tigra novel hub & lid methods for dynamic graph embeddings based on temporal random walks
"""

from tigra.core.dgraph import TemporalAggregateGraph

from tigra.embeddings.rndw.trndwbase import TemporalRWEmbedding
from tigra.embeddings.rndw.trndwbase import TempRWSampler

from tigra.evaluation.lideval import NCLIDEstimator

import numpy as np
import math
import random


class LIDTempRandomWalk(TempRWSampler):
    def __init__(
        self, graph: TemporalAggregateGraph, 
        num_walks=10, 
        walk_length=32, 
        num_workers=1, 
        inverse=False,
        log_scaling=False,
        u=0,
    ):
        super().__init__(graph)
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers
        self.inverse = inverse
        self.log_scaling = log_scaling
        self.u = u

        self.lidestimator = NCLIDEstimator(graph.G)
        self.lidestimator.estimate_lids()
        self.lids = self.lidestimator.get_lid_values()


    def select_next_node(self, start_node, current_node, prev_node, active_neighbours):
        r = random.random()
        if r < self.u:
            return random.sample(active_neighbours, 1)[0]
        
        probabilities = [0.0] * len(active_neighbours)

        max_lid = 0.0
        for n in active_neighbours:
            if self.lids[n] > max_lid:
                max_lid = self.lids[n]

        if max_lid == 0:
            return np.random.choice(active_neighbours, 1)[0]

        for i in range(len(active_neighbours)):
            n = active_neighbours[i]
            lid = self.lids[n]
            if self.inverse:
                probabilities[i] = 1 + max_lid - lid
            else:
                probabilities[i] = 1 + lid

        if self.log_scaling:
            for i in range(len(active_neighbours)):
                probabilities[i] = math.log(1 + probabilities[i])

        total = 0.0
        for p in probabilities:
            total += p

        if total == 0.0:
            return np.random.choice(active_neighbours, 1)[0]

        probs = [p / total for p in probabilities]
        return np.random.choice(active_neighbours, 1, p=probs)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class DeepLIDTemporalWalk(TemporalRWEmbedding):
    def __init__(self, 
        dim, 
        num_walks=10, 
        walk_length=32, 
        num_workers=1,
        inverse=False,
        log_scaling=False,
        u=0
    ):
        super().__init__(dim, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.inverse = inverse
        self.log_scaling = log_scaling
        self.u = u
        
    def create_temprw_sampler(self, graph: TemporalAggregateGraph) -> TempRWSampler:
        return LIDTempRandomWalk(
            graph, 
            num_walks=self.num_walks, 
            walk_length=self.walk_length, 
            num_workers=self.num_workers,
            inverse=self.inverse,
            log_scaling=self.log_scaling,
            u = self.u
        )
    


class HubTempRandomWalk(TempRWSampler):
    def __init__(
        self, graph: TemporalAggregateGraph, 
        num_walks=10, 
        walk_length=32, 
        num_workers=1, 
        inverse=False,
        log_scaling=False,
        u=0,
    ):
        super().__init__(graph)
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers
        self.inverse = inverse
        self.log_scaling = log_scaling
        self.u = u


    def select_next_node(self, start_node, current_node, prev_node, active_neighbours):
        r = random.random()
        if r < self.u:
            return random.sample(active_neighbours, 1)[0]
        
        probabilities = [0.0] * len(active_neighbours)
        degrees = [0] * len(active_neighbours)

        max_deg = 0.0
        for i in range(len(active_neighbours)):
            n = active_neighbours[i]
            deg = self.graph.G.degree[n]
            degrees[i] = deg
            if deg > max_deg:
                max_deg = deg

        if max_deg == 0:
            return np.random.choice(active_neighbours, 1)[0]

        for i in range(len(active_neighbours)):
            n = active_neighbours[i]
            deg = degrees[i]
            if self.inverse:
                probabilities[i] = 1 + max_deg - deg
            else:
                probabilities[i] = 1 + deg

        if self.log_scaling:
            for i in range(len(active_neighbours)):
                probabilities[i] = math.log(1 + probabilities[i])

        total = 0.0
        for p in probabilities:
            total += p

        if total == 0.0:
            return np.random.choice(active_neighbours, 1)[0]

        probs = [p / total for p in probabilities]
        return np.random.choice(active_neighbours, 1, p=probs)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl
    

class DeepHubTemporalWalk(TemporalRWEmbedding):
    def __init__(self, 
        dim, 
        num_walks=10, 
        walk_length=32, 
        num_workers=1,
        inverse=False,
        log_scaling=False,
        u=0
    ):
        super().__init__(dim, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.inverse = inverse
        self.log_scaling = log_scaling
        self.u = u
        
    def create_temprw_sampler(self, graph: TemporalAggregateGraph) -> TempRWSampler:
        return HubTempRandomWalk(
            graph, 
            num_walks=self.num_walks, 
            walk_length=self.walk_length, 
            num_workers=self.num_workers,
            inverse=self.inverse,
            log_scaling=self.log_scaling,
            u = self.u
        )