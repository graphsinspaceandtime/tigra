"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Main base class for dynamic/temporal graph embeddings
"""

from abc import ABC, abstractmethod
from typing import List


class SingletonEmbedding:
    """
    Embedding of one snapshot or an aggregate dynamic graph representation
    """

    def __init__(self):
        self.embedding = dict()

    def add_embedding_vector(self, id, vector: List[float]):
        self.embedding[id] = vector

    def __str__(self):
        s = ""
        for key, value in sorted(self.embedding.items()):
            s += str(key) + " --> " + str([round(val, 2) for val in value]) + "\n"

        return s


class Embedding(ABC):
    """
    Base class for dynamic/temporal graph embeddings
    """

    def __init__(self, dim):
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
    def construct(self, data) -> List[SingletonEmbedding]:
        """
        Embedding algorithm that must be implemented in subclasses.
        """
        pass

    def make_embedding(self, data):
        self.embeddings = self.construct(data)
