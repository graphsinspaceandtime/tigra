"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning#
in Complex Dynamical Systems

Factory for SOTA dynamic graph embedding methods

"""

from abc import ABC, abstractmethod
import typing

from data.bank import Databank

from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.embedding import Embedding


class Factory(ABC):
    def __init__(self, datapath=None):
        if datapath == None:
            databank = Databank()
        else:
            databank = Databank(datapath)

    @abstractmethod
    def get_embedding(self, dataset: str) -> Embedding:
        pass


class DN2V_Factory(Factory):
    def __init__(self, dn2vconf, datapath=None):
        super().__init__(datapath)

        conf_f = open(dn2vconf, "r")
        conf_f.readline()  # header

        self.configs = dict()

        for line in conf_f:
            tokens = line.split(",")
            dsname = tokens[0]
            dim = int(tokens[1])
            p = float(tokens[2])
            q = float(tokens[3])
            walk_len = int(tokens[4])
            num_walks = int(tokens[5])
            cumulative = True if tokens[6] == "True" else False

            self.configs[dsname] = (dim, p, q, walk_len, num_walks, cumulative)

    def get_embedding(self, dataset: str) -> Embedding:
        if not dataset in self.configs:
            print("WARNING: dataset not in dn2v configuration")
            return None  # type: ignore

        conf = self.configs[dataset]

        dn2v = Dynnode2vec(
            conf[0],
            p=conf[1],
            q=conf[2],
            walk_length=conf[3],
            n_walks_per_node=conf[4],
            parallel_processes=16,
        )

        es, time_resolution = load_event_sequence(dataset)  # type: ignore

        dg = DynamicGraph()
        dg.init_from_event_sequence(
            es, time_resolution=time_resolution, directed=False, cumulative=conf[5]
        )

        dn2v.make_embedding(dg)

        return dn2v, dg  # type: ignore


dn2v_factory = None


def get_dn2v_factory(dn2vconf, datapath=None):
    global dn2v_factory

    if dn2v_factory == None:
        dn2v_factory = DN2V_Factory(dn2vconf, datapath=datapath)

    return dn2v_factory
