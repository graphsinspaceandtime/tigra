"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

LID-based classes for dynamic graph embedding methods based on random walks

Author: lucy@dmi.uns.ac.rs
"""


from tigra.embeddings.rndw.rndwbase import RandomWalkSampler
from tigra.embeddings.rndw.rndwbase import RWEmbBase
from tigra.embeddings.rndw.rndwbase import PQRandomWalk
from tigra.embeddings.rndw.rndwbase import RWEmbEgoDelta
from tigra.evaluation.lideval import NCLIDEstimator

import networkx as nx
import math
import random
from typing import List


class RWEmbLIDDelta(RWEmbBase):
    def __init__(self, dim, delta_nodes=True, num_workers=1):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        
    def determine_delta_nodes(self, graph: nx.Graph, prev_graph: nx.Graph) -> List[int]:
        lidestimator = NCLIDEstimator(graph)
        lidestimator.estimate_lids()
        lids_curr = lidestimator.get_lid_values()
        
        lidestimator = NCLIDEstimator(prev_graph)
        lidestimator.estimate_lids()
        lids_prev = lidestimator.get_lid_values()
        
        nodes_with_modified_lids = set()
        new_nodes = set()
        if prev_graph is not None:
            for n in graph.nodes:
                if n not in prev_graph.nodes():
                    new_nodes.add(n)
                else:
                    if lids_curr[n] != lids_prev[n]:
                        nodes_with_modified_lids.add(n)
        
        delta_nodes = list(new_nodes | nodes_with_modified_lids)
        return delta_nodes
    

class DynN2V_LIDDelta(RWEmbLIDDelta):
    def __init__(
        self, dim, p, q, delta_nodes=True, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.p = p
        self.q = q

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return PQRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )


class LIDRandomWalk(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)
        lidestimator = NCLIDEstimator(graph)
        lidestimator.estimate_lids()
        self.lids_curr = lidestimator.get_lid_values()

    def num_walks(self, node):
        nw = math.ceil((1 + self.lids_curr[node]) * self.nw)
        return nw

    def walk_length(self, node):
        wl = math.ceil(self.wl / (1 + self.lids_curr[node]))
        return wl


class DynN2V_LIDRW(RWEmbEgoDelta):
    def __init__(
        self, dim, p, q, delta_nodes=True, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.p = p
        self.q = q

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return LIDRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )


class DynN2V_LIDRWDelta(RWEmbLIDDelta):
    def __init__(
        self, dim, p, q, delta_nodes=True, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.p = p
        self.q = q

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return LIDRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )




'''
class LIDAwareRandomWalk(RandomWalkSampler):    
    def __init__(
        self, graph: nx.Graph, prev_graph, nodes, num_workers=1
    ):
        super().__init__(graph, nodes)
        self.B = 10
        self.W = 32 #80
        self.num_workers = num_workers
        
        lidestimator = NCLIDEstimator(graph)
        lidestimator.estimate_lids()
        self.lids_curr = lidestimator.get_lid_values()
        
        self.lids_prev = None
        if prev_graph is not None:
            lidestimator = NCLIDEstimator(prev_graph)
            lidestimator.estimate_lids()
            self.lids_prev = lidestimator.get_lid_values()
        

    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        if self.lids_prev is None:
            return random.sample(neighbours, 1)[0]
        lid_changed = []
        for n in neighbours:
            if self.lids_curr[n] != self.lids_prev[n]:
                lid_changed.append(n)
        return random.sample(lid_changed, 1)[0]

    def num_walks(self, node):
        self.nw = math.ceil((1 + self.lids_curr[node]) * self.B)
        return self.nw

    def walk_length(self, node):
        self.wl = math.ceil(self.W / (1 + self.lids_curr[node]))
        return self.wl
'''    
    
    
