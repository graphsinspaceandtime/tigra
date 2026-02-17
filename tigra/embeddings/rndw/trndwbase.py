"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Base classes for dynamic graph embedding methods based on temporal random walks
"""

from abc import abstractmethod

import random
from multiprocessing import Pool
import typing
from typing import List

from gensim.models import Word2Vec

from tigra.core.dgraph import TemporalAggregateGraph
from tigra.embeddings.embedding import Embedding, SingletonEmbedding


class TempRWSampler:
    """
    Base class for temporal random walk samplers
    """

    def __init__(self, graph: TemporalAggregateGraph, num_workers=1):
        self.graph = graph
        self.num_workers = num_workers

    @abstractmethod
    def select_next_node(self, start_node, current_node, prev_node, active_neighbours):
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
        current_time = self.graph.G.nodes[start_node]["timestamp"]
        
        #dbg_str = str(start_node) + "[" + str(current_time) + "]"

        while len(walk) < walk_length:
            current_node = walk[-1]
            current_neis = list(self.graph.G.neighbors(current_node))
            if len(current_neis) == 0:
                break

            # select active neighbours w.r.t current_time
            active_neis = []
            for nei in current_neis:
                timestamps = self.graph.G[current_node][nei]["edge_data"]
                maxt = max(timestamps)
                if current_time <= maxt:
                    active_neis.append(nei)

            if len(active_neis) == 0:
                break

            next_node = self.select_next_node(
                start_node, current_node, prev_node, active_neis
            )
            if next_node == None:
                break

            walk.append(next_node)

            # update time
            timestamps = self.graph.G[current_node][next_node]["edge_data"]
            for t in timestamps:
                if t > current_time:
                    current_time = t
                    break

            #dbg_str += str(next_node) + "[" + str(current_time) + "]"

            prev_node = current_node

        #print(dbg_str)
        return walk

    def sample_rnd_walks_from_node(self, node):
        walks = []
        num_walks = self.num_walks(node)
        walk_length = self.walk_length(node)
        for _ in range(num_walks):
            walk = self.rnd_walks_from_node(walk_length=walk_length, start_node=node)
            walks.append(walk)

        return walks

    def sample_walks(self):
        walks = []
        if self.num_workers == 1:
            for node in self.graph.G.nodes():
                w = self.sample_rnd_walks_from_node(node)
                walks.extend(w)
        else:
            pool = Pool(self.num_workers)
            wlk = pool.map(func=self.sample_rnd_walks_from_node, iterable=self.graph.G.nodes())
            walks = [w for node_walks in wlk for w in node_walks]

        random.shuffle(walks)
        return walks
    

class UnbiasedTempRandomWalk(TempRWSampler):
    def __init__(
        self, graph: TemporalAggregateGraph, num_walks=10, walk_length=32, num_workers=1
    ):
        super().__init__(graph)
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers

    def select_next_node(self, start_node, current_node, prev_node, active_neighbours):
        return random.sample(active_neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl
    

class TemporalRWEmbedding(Embedding):
    def __init__(self, dim, num_workers=1):
        super().__init__(dim)
        self.num_workers = num_workers

    def construct(self, data) -> List[typing.Union[SingletonEmbedding, None]]:
        if not isinstance(data, TemporalAggregateGraph):
            raise Exception(
                "{} works with TemporalAggregateGraph objects.".format(type(self).__name__)
            )

        trw_sampler = self.create_temprw_sampler(data)
        walks = trw_sampler.sample_walks()
        
        model = Word2Vec(
            sentences=walks,
            vector_size=self.dim,
            window=5,
            min_count=0,
            sg=1,
            workers=self.num_workers
        )

        emb = SingletonEmbedding()
        for node in data.G.nodes():
            vector = model.wv[node].tolist()
            emb.add_embedding_vector(node, vector)

        return [emb]

    @abstractmethod
    def create_temprw_sampler(self, graph: TemporalAggregateGraph) -> TempRWSampler:
        pass


class DeepTemporalWalk(TemporalRWEmbedding):
    def __init__(self, dim, num_walks=10, walk_length=32, num_workers=1):
        super().__init__(dim, num_workers=num_workers)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        
    def create_temprw_sampler(self, graph: TemporalAggregateGraph) -> TempRWSampler:
        return UnbiasedTempRandomWalk(graph, num_walks=self.num_walks, walk_length=self.walk_length, num_workers=self.num_workers)


