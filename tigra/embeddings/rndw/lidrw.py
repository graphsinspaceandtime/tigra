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
import numpy as np
import bisect


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


class RWEmbNCDelta(RWEmbBase):
    def __init__(self, dim, delta_nodes=True, num_workers=1):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)

    def determine_delta_nodes(self, graph: nx.Graph, prev_graph: nx.Graph) -> List[int]:
        lide = NCLIDEstimator(graph)
        lide.estimate_lids()

        lide_prev = NCLIDEstimator(prev_graph)
        lide_prev.estimate_lids()

        nodes_with_modified_nc = set()
        new_nodes = set()

        for n in graph.nodes:
            if n not in prev_graph.nodes():
                new_nodes.add(n)
            else:
                nc = set(lide.natural_community[n])
                prev_nc = set(lide_prev.natural_community[n])
                if nc != prev_nc:
                    nodes_with_modified_nc.add(n)

        delta_nodes = list(new_nodes | nodes_with_modified_nc)
        return delta_nodes


class RWEmbHLIDDelta(RWEmbBase):
    def __init__(self, dim, delta_nodes=True, num_workers=1):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)

    def determine_delta_nodes(self, graph: nx.Graph, prev_graph: nx.Graph) -> List[int]:
        lide = NCLIDEstimator(graph)
        lide.estimate_lids()
        lids = lide.get_lid_values()
        # avg_lid = np.array(list(lids.values())).mean()

        new_nodes = set()
        hlid_nodes = set()
        ego_nodes = set()

        for n in graph.nodes:
            if n not in prev_graph.nodes():
                new_nodes.add(n)
            else:
                lidv = lids[n]
                if lidv > 0:
                    # if lidv > avg_lid:
                    hlid_nodes.add(n)

                curr_n = set(graph.neighbors(n))
                prev_n = set(prev_graph.neighbors(n))
                if curr_n != prev_n:
                    ego_nodes.add(n)

        """
        print("NEW NODES", new_nodes)
        print("HLID NODES", hlid_nodes)
        print("EGO NODES", ego_nodes)
        print(len(new_nodes), len(hlid_nodes), len(ego_nodes))
        print(len(hlid_nodes & ego_nodes))
        print("SUBSET", hlid_nodes <= ego_nodes)
        print()
        """

        delta_nodes = list(new_nodes | hlid_nodes | ego_nodes)
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


class DynN2V_NCDelta(RWEmbNCDelta):
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


class DynN2V_HLIDDelta(RWEmbHLIDDelta):
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
        self,
        graph: nx.Graph,
        nodes,
        p,
        q,
        num_walks=10,
        walk_length=32,
        num_workers=1,
        scale_factor=1.0,
        boost=False,
    ):
        super().__init__(
            graph,
            nodes,
            p,
            q,
            num_walks=num_walks,
            walk_length=walk_length,
            num_workers=num_workers,
        )

        lidestimator = NCLIDEstimator(graph)
        lidestimator.estimate_lids()
        self.lids = lidestimator.get_lid_values()

        self.wl_per_node = dict()
        self.nw_per_node = dict()

        total_steps = num_walks * walk_length

        for node in nodes:
            """
            self.wl_per_node[node] = math.ceil(
                self.wl / (scale_factor * (1 + self.lids[node]))
            )

            self.nw_per_node[node] = math.ceil(
                (1 + self.lids[node]) * self.nw * scale_factor
            )
            """

            new_nw, new_wl = num_walks, walk_length
            if self.lids[node] > 0:
                new_nw = math.ceil(num_walks * self.lids[node] * scale_factor)
                if not boost:
                    new_wl = math.ceil(total_steps / new_nw)

            self.wl_per_node[node] = new_wl
            self.nw_per_node[node] = new_nw

        """
        print("NODE,NW=10,WL=32")
        for node in nodes:
            print(node, self.nw_per_node[node], self.wl_per_node[node])
        print()
        """

    def num_walks(self, node):
        return self.nw_per_node[node]

    def walk_length(self, node):
        return self.wl_per_node[node]


class DynN2V_LIDRW(RWEmbEgoDelta):
    def __init__(
        self,
        dim,
        p,
        q,
        delta_nodes=True,
        num_walks=10,
        walk_length=32,
        num_workers=1,
        scale_factor=1.0,
        boost=False,
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.p = p
        self.q = q
        self.scale_factor = scale_factor
        self.boost = boost

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return LIDRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
            scale_factor=self.scale_factor,
            boost=self.boost,
        )


class DynN2V_LIDRWDelta(RWEmbLIDDelta):
    def __init__(
        self,
        dim,
        p,
        q,
        delta_nodes=True,
        num_walks=10,
        walk_length=32,
        num_workers=1,
        scale_factor=1.0,
        boost=False,
    ):
        super().__init__(dim, delta_nodes=delta_nodes, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.p = p
        self.q = q
        self.scale_factor = scale_factor
        self.boost = boost

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        return LIDRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
            scale_factor=self.scale_factor,
            boost=self.boost,
        )


class WLLIDRandomWalk(PQRandomWalk):
    """
    Walk length adjusted according to NC-LID
    """

    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(
            graph,
            nodes,
            p,
            q,
            num_walks=num_walks,
            walk_length=walk_length,
            num_workers=num_workers,
        )
        lidestimator = NCLIDEstimator(graph)
        lidestimator.estimate_lids()
        self.lids = lidestimator.get_lid_values()

        self.wl_per_node = dict()
        total_wl = len(nodes) * walk_length

        # Laplace smoothing
        total_lid = sum(self.lids.values()) + len(nodes)
        lid_norm = {node: (self.lids[node] + 1.0) / total_lid for node in self.nodes}
        for node in lid_norm:
            lid_norm_value = lid_norm[node]
            lid_wl = int(round(total_wl * lid_norm_value))
            self.wl_per_node[node] = lid_wl

        print("WL_PER_NODE", self.wl_per_node)

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl_per_node[node]


class DynN2V_WLLID(RWEmbEgoDelta):
    """
    DynN2V modification in which walk length is adjusted according to NC-LID
    """

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
        return WLLIDRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )


class NWLIDRandomWalk(PQRandomWalk):
    """
    Num walks adjusted according to NC-LID
    """

    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(
            graph,
            nodes,
            p,
            q,
            num_walks=num_walks,
            walk_length=walk_length,
            num_workers=num_workers,
        )
        lidestimator = NCLIDEstimator(graph)
        lidestimator.estimate_lids()
        self.lids = lidestimator.get_lid_values()

        self.nw_per_node = dict()
        total_nw = len(nodes) * num_walks

        # Laplace smoothing
        total_lid = sum(self.lids.values()) + len(nodes)
        lid_norm = {node: (self.lids[node] + 1.0) / total_lid for node in self.nodes}
        for node in lid_norm:
            lid_norm_value = lid_norm[node]
            lid_nw = int(round(total_nw * lid_norm_value))
            self.nw_per_node[node] = lid_nw

        # print("NW_PER_NODE", self.nw_per_node)

    def num_walks(self, node):
        return self.nw_per_node[node]

    def walk_length(self, node):
        return self.wl


class DynN2V_NWLID(RWEmbEgoDelta):
    """
    DynN2V modification in which walk length is adjusted according to NC-LID
    """

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
        return NWLIDRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_workers=self.num_workers,
        )


class PQLIDRandomWalk(RandomWalkSampler):
    """
    P and Q adjusted according to NC-LID
    """

    def __init__(
        self, graph: nx.Graph, nodes, p, q, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph, nodes)
        self.p = p
        self.q = q
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers

        self.lidestimator = NCLIDEstimator(graph)
        self.lidestimator.estimate_lids()
        self.lids = self.lidestimator.get_lid_values()

        self.compute_transition_probabilities()

    def compute_transition_probabilities(self):
        self.transition_weight = dict()

        ip, iq = 1.0 / self.p, 1.0 / self.q

        for node in self.G.nodes():
            neis = list(self.G.neighbors(node))

            for previous_node in neis:
                previous_node_neis = list(self.G.neighbors(previous_node))
                weights = np.ones(len(neis))

                for i in range(len(neis)):
                    n = neis[i]
                    if n == previous_node:
                        if self.lidestimator.is_in_natural_community(node, n):
                            weights[i] *= ip
                        else:
                            weights[i] *= 1 / (self.p + self.lids[n])

                    if n not in previous_node_neis:
                        if self.lidestimator.is_in_natural_community(node, n):
                            weights[i] *= iq
                        else:
                            weights[i] *= 1 / (self.q + self.lids[n])

                self.transition_weight[(node, previous_node)] = weights

            # special case, start node in the walk (previous_node == None)
            self.transition_weight[(node, None)] = np.ones(len(neis))

    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        weights = self.transition_weight[(current_node, prev_node)]
        probs = np.cumsum(weights)
        total = probs[-1]
        index = bisect.bisect(probs, random.random() * total)
        # print(len(neighbours), len(weights), index)
        ret = neighbours[index]
        if ret == None:
            print("[PQLIDRandomWalk], select next node error")

        return ret

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class DynN2V_PQLID(RWEmbEgoDelta):
    """
    DynN2V modification in which p & q are adjusted according to NC-LID
    """

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
        return PQLIDRandomWalk(
            graph,
            nodes,
            self.p,
            self.q,
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


class DynDeepLIDWalk(RWEmbEgoDelta):
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


"""
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
"""
"""
        probabilities = []
        total_lid, max_lid = 0, 0

        for n in neighbours:
            lid = self.lids[n]
            total_lid += lid
            if lid > max_lid:
                max_lid = lid

        if self.inverse:
            total_lid = 0
            for n in neighbours:
                total_lid += max_lid - self.lids[n]

        if total_lid == 0:
            return np.random.choice(neighbours, 1)[0]

        if self.log_scaling:


        for n in neighbours:
            if not self.inverse:
                probabilities.append(self.lids[n]/total_lid)
            else:
                probabilities.append((max_lid - self.lids[n])/total_lid)
"""
