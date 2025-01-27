"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

LID-based classes for dynamic graph embedding methods based on random walks
"""


from tigra.embeddings.rndw.rndwbase import RandomWalkSampler
from tigra.embeddings.rndw.rndwbase import RWEmbBase
from tigra.evaluation.lideval import NCLIDEstimator

import networkx as nx
import math
import random
from typing import List


'''
lidestimator = NCLIDEstimator(graph.G)
        lidestimator.estimate_lids()
        lid_vals = lidestimator.get_lid_values()
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
    
    
    
class LIDAwareEmbedding(RWEmbBase):
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
        if prev_graph is not None:
            for n in graph.nodes:
                if n in lids_prev.keys() and lids_curr[n] != lids_prev[n]:
                    nodes_with_modified_lids.add(n)
        delta_nodes = list(graph.nodes & nodes_with_modified_lids)
        return delta_nodes
    
    
    def create_random_walk_sampler(self, graph: nx.Graph, nodes) -> RandomWalkSampler:
        return LIDAwareRandomWalk(graph, None, nodes)