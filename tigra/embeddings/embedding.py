"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Main base class for dynamic/temporal graph embeddings
"""

from abc import ABC, abstractmethod
from typing import List
import typing
from tigra.core.dgraph import Graph

import numpy as np
import heapq

import os, shutil


class SingletonEmbedding:
    """
    Embedding of one snapshot or an aggregate dynamic graph representation
    """

    def __init__(self):
        self.embedding = dict()
        self.dists = dict()
        self.label = dict()

    def add_embedding_vector(self, id, vector: List[float]):
        self.embedding[id] = vector

    def __str__(self):
        s = ""
        for key, value in sorted(self.embedding.items()):
            s += str(key) + " --> " + str([round(val, 2) for val in value]) + "\n"

        return s

    def get_dist(self, node1, node2, save=False):
        if node1 > node2:
            node1, node2 = node2, node1

        if node1 in self.dists and node2 in self.dists[node1]:
            return self.dists[node1][node2]

        n1emb = np.array(self.embedding[node1])
        n2emb = np.array(self.embedding[node2])
        dist = np.linalg.norm(n1emb - n2emb)
        if save:
            if not node1 in self.dists:
                self.dists[node1] = {}
            self.dists[node1][node2] = dist

        return dist

    def reconstruct(self, k, name, cache_dists=False):
        """
        Reconstructs the graph from its embedding.
        Important: graph reconstruction always produces undirected graph,
            so before comparison convert directed graphs to undirected

        Parameters
        ----------
        k : int
            The number of the links in the resulting reconstructed graph.
        cache_dists : boolean
            If true, the pairwise distances will be cached for later use.

        Returns
        ----------
        - core.dgraph.Graph - reconstructed graph.
        """

        nodes = list(self.embedding.keys())
        dists = []
        for i in range(len(nodes)):
            node1 = nodes[i]
            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                dist = self.get_dist(node1, node2, cache_dists)
                dists.append(
                    (
                        dist,
                        node1,
                        node2,
                    )
                )
        heapq.heapify(dists)

        g = Graph(name, directed=False)
        for node in nodes:
            g.add_node(node)
        for edge in heapq.nsmallest(k, dists):
            g.add_edge(edge[1], edge[2])

        return g

    def __getitem__(self, node):
        """
        Returns the embedding at given index
        """
        return self.embedding[node]

    def save_to_file(self, file: str):
        """
        Saves embeddings to a csv file.
        """
        with open(file, "w", encoding="utf-8") as f:
            for id, vector in self.embedding.items():
                f.write(str(id))
                f.write(",")
                f.write(",".join([str(v) for v in vector]))
                f.write("\n")

    def load_from_file(self, file: str):
        """
        Loads embeddings from a csv file. It needs to be formatted in the same
        way as saved file.
        """
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split(",")
                self.add_embedding_vector(
                    int(line[0]), [float(point) for point in line[1:]]
                )

    def load_labels_from_file(self, file: str):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split(",")
                node_id = int(line[0])
                label = line[1].strip()
                self.label[node_id] = label

    def get_labels(self):
        return self.label

    def get_node_ids(self):
        return list(self.embedding.keys())

    def get_label(self, node):
        return self.label[node]


class Embedding(ABC):
    """
    Base class for dynamic/temporal graph embeddings
    """

    dim: int

    def __init__(self, dim: int):
        self.embeddings = []
        self.dim = dim

    def num_embeddings(self):
        return len(self.embeddings)

    def is_singleton_embedding(self):
        return len(self.embeddings) == 1

    def __getitem__(self, index):
        """
        Returns the embedding at given index
        """
        return self.embeddings[index]

    @abstractmethod
    def construct(self, data) -> List[typing.Union[SingletonEmbedding, None]]:
        """
        Embedding algorithm that must be implemented in subclasses.
        """
        pass

    def make_embedding(self, data):
        self.embeddings = self.construct(data)

    def save_embedding(self, folder_name: str):
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)

        os.mkdir(folder_name)
        for i in range(len(self.embeddings)):
            fn = str(i) + ".emb"
            path = os.path.join(folder_name, fn)
            self.embeddings[i].save_to_file(path)


class FSEmbedding(Embedding):
    """
    Embeddings loadead from file system
    """

    def __init__(self, folder_name: str):
        super().__init__(0)

        if not os.path.exists(folder_name):
            raise Exception(folder_name + " does not exist")

        onlyfiles = [
            f
            for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f))
        ]

        embs = [f for f in onlyfiles if f.endswith(".emb")]
        embs.sort()
        # print(embs)
        for emb in embs:
            s = SingletonEmbedding()
            path = os.path.join(folder_name, emb)
            s.load_from_file(path)
            self.embeddings.append(s)

        labs = [f for f in onlyfiles if f.endswith(".labels")]
        if len(labs) == 0:
            return

        labs.sort()
        for i in range(len(labs)):
            lab = labs[i]
            emb = self.embeddings[i]
            path = os.path.join(folder_name, lab)
            emb.load_labels_from_file(path)

    def construct(self, data) -> List[typing.Union[SingletonEmbedding, None]]:
        raise Exception("This method should not be called for FSEmbedding instances")
