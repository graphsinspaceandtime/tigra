"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

HUB-based classes for dynamic graph embedding methods based on random walks
"""


from tigra.embeddings.rndw.rndwbase import RWEmbBase, RandomWalkSampler, RWEmbEgoDelta, PQRandomWalk

import networkx as nx
import numpy as np
import math

from typing import List


class HubChangedEmbedding(RWEmbBase):
    def __init__(self, dim, delta_nodes=True, num_workers=1):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)

    def determine_delta_nodes(self, graph: nx.Graph, prev_graph: nx.Graph) -> List[int]:
        nodes = []
        nodes_with_modified_degree = []

        if prev_graph is not None:
            for n in graph.nodes:
                if n not in prev_graph.nodes:
                    nodes.append(n)
                else:
                    if graph.degree[n] != prev_graph.degree[n]:
                        nodes_with_modified_degree.append(n)
        
        delta_nodes = list(graph.nodes | nodes_with_modified_degree) 
        return delta_nodes      



class HubSampler(RandomWalkSampler):
    def __init__(
        self, graph: nx.Graph, nodes, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes)
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers

    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        probabilities = []
        total_hub = 0

        for n in neighbours:
            total_hub += self.G.degree[n]

        for n in neighbours:
            probabilities.append(self.G.degree[n] / total_hub)


        probabilities = np.asarray(probabilities).astype('float64')
        probabilities = probabilities / np.sum(probabilities)

        return np.random.choice(neighbours, 1, p = probabilities)[0]       
        

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl

        

class HubSamplerReverse(RandomWalkSampler):
    def __init__(
        self, graph: nx.Graph, nodes, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes)
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers

    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        probabilities = []
        total_hub = 0

        for n in neighbours:
            total_hub += self.G.degree[n]

        for n in neighbours:
            probabilities.insert(0, self.G.degree[n] / total_hub)


        probabilities = np.asarray(probabilities).astype('float64')
        probabilities = probabilities / np.sum(probabilities)

        return np.random.choice(neighbours, 1, p = probabilities)[0]       
        

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


"""
 Base class when num_walks and/or walk_length change
 Change num_walks and/or walk_length as you prefer
"""
class HubnessRandomWalk(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)

    def num_walks(self, node):
        nw = math.ceil((1 + self.G.degree[node]) * self.nw)
        return nw

    def walk_length(self, node):
        wl = math.ceil(self.wl / (1 + self.G.degree[node]))
        return wl



class HubnessRandomWalkInverseDegree(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)

    def num_walks(self, node):
        nw = math.ceil(((1 / self.G.degree[node])) * self.nw)
        return nw

    def walk_length(self, node):
        wl = math.ceil(self.wl / (1 / self.G.degree[node]))
        return wl



class HubnessRandomWalkLogDegree(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)

    def num_walks(self, node):
        nw = math.ceil((1 + math.log(self.G.degree[node])) * self.nw)
        return nw

    def walk_length(self, node):
        wl = math.ceil(1 + math.log(self.G.degree[node]))
        return wl


class HubnessRandomWalkMaxDegree(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)

        self.degrees = []

        for node in nodes:
            self.degrees.append(self.G.degree[node])
    
    def num_walks(self, node):
        return max(self.degrees)

    def walk_length(self, node):
        return max(self.degrees)



class HubnessRandomWalkMaxLogDegree(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)

        self.log_degrees = []

        for node in nodes:
            self.log_degrees.append(1 + (math.log(self.G.degree[node] if self.G.degree[node] > 0 else 0)))
    
    def num_walks(self, node):
        return math.ceil(max(self.log_degrees))

    def walk_length(self, node):
        return math.ceil(max(self.log_degrees))



class HubnessRandomWalkInverseLogDegree(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)

        self.log_degrees_dict = dict()

        for node in nodes:
            self.log_degrees_dict[node] = (1 + (math.log(self.G.degree[node] if self.G.degree[node] > 0 else 0)))
    
    def num_walks(self, node):
        nw = math.ceil(((1 / self.log_degrees_dict[node])) * self.nw)
        return nw

    def walk_length(self, node):
        wl = math.ceil(self.log_degrees_dict[node])
        return wl



class HubnessRandomWalkSqrtDegree(PQRandomWalk):
    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes, p, q, num_walks=num_walks, walk_length=walk_length, num_workers=num_workers)

    def num_walks(self, node):
        nw = math.ceil(math.sqrt(self.G.degree[node]))
        return nw

    def walk_length(self, node):
        wl = math.ceil(math.sqrt(self.G.degree[node]))
        return wl



class HubRandomWalk(RWEmbEgoDelta):
    def __init__(
        self, dim, delta_nodes=True, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
    

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return HubSampler(
            graph,
            nodes,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )


class HubChangedRandomWalk(HubChangedEmbedding):
    def __init__(
        self, dim, delta_nodes=True, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
    

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return HubSampler(
            graph,
            nodes,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )


class HubChangedRandomWalkReverse(HubChangedEmbedding):
    def __init__(
        self, dim, delta_nodes=True, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
    

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return HubSamplerReverse(
            graph,
            nodes,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )



class DynN2V_HubDelta(HubChangedEmbedding):
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



class DynN2V_HubDeltaNWWL(HubChangedEmbedding):
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
        return HubnessRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )



class DynN2V_HubDeltaInverseDegree(HubChangedEmbedding):
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
        return HubnessRandomWalkInverseDegree(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )



class DynN2V_HubDeltaLogDegree(HubChangedEmbedding):
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
        return HubnessRandomWalkLogDegree(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )



class DynN2V_HubDeltaMaxDegree(HubChangedEmbedding):
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
        return HubnessRandomWalkMaxDegree(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )



class DynN2V_HubDeltaMaxLogDegree(HubChangedEmbedding):
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
        return HubnessRandomWalkMaxLogDegree(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )



class DynN2V_HubDeltaInverseLogDegree(HubChangedEmbedding):
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
        return HubnessRandomWalkInverseLogDegree(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )



class DynN2V_HubDeltaSqrtDegree(HubChangedEmbedding):
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
        return HubnessRandomWalkSqrtDegree(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )