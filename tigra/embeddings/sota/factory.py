"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning
in Complex Dynamical Systems

Factory for SOTA dynamic graph embedding methods

"""

from abc import ABC, abstractmethod
import typing

from data.bank import Databank

# from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.embeddings.rndw.rndwbase import DynN2V
from tigra.embeddings.rndw.hubrw import DynDeepHubWalk
from tigra.embeddings.sota.stwalk import STWalk
from tigra.embeddings.sota.ctdne_wrapper import Ctdne
from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicGraph
from tigra.core.dgraph import DynamicAggregateGraph
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

        """
        dn2v = Dynnode2vec(
            conf[0],
            p=conf[1],
            q=conf[2],
            walk_length=conf[3],
            n_walks_per_node=conf[4],
            parallel_processes=16,
        )
        """
        dn2v = DynN2V(conf[0], conf[1], conf[2], num_walks=conf[4], walk_length=conf[3])

        es, time_resolution = load_event_sequence(dataset)  # type: ignore

        dg = DynamicGraph()
        dg.init_from_event_sequence(
            es, time_resolution=time_resolution, directed=False, cumulative=conf[5]
        )

        dn2v.make_embedding(dg)

        return dn2v, dg  # type: ignore


class STWalk_Factory(Factory):
    def __init__(self, stwalkconf, datapath=None):
        super().__init__(datapath)

        conf_f = open(stwalkconf, "r")
        conf_f.readline()  # header

        self.configs = dict()

        for line in conf_f:
            tokens = line.split(",")
            dsname = tokens[0]
            dim = int(tokens[1])
            timewindow = int(tokens[2])
            cumulative = True if tokens[3] == "True" else False

            self.configs[dsname] = (dim, timewindow, cumulative)

    def get_embedding(self, dataset: str) -> Embedding:
        if not dataset in self.configs:
            print("WARNING: dataset not in stwalk configuration")
            return None  # type: ignore

        conf = self.configs[dataset]

        es, time_resolution = load_event_sequence(dataset)  # type: ignore

        dg = DynamicGraph()
        dg.init_from_event_sequence(
            es, time_resolution=time_resolution, directed=False, cumulative=conf[2]
        )
        
        stw = STWalk(number_restart=10, walk_length=32, dimension=conf[0], time_step=len(dg.get_graph_sequence()), time_window_size=conf[1],workers=16,vocab_window_size=5)
       

        stw.make_embedding(dg)

        return stw, dg  # type: ignore


class CTDNE_Factory(Factory):
    def __init__(self, ctdneconf, datapath=None):
        super().__init__(datapath)

        conf_f = open(ctdneconf, "r")
        conf_f.readline()  # header

        self.configs = dict()

        for line in conf_f:
            tokens = line.split(",")
            dsname = tokens[0]
            dim = int(tokens[1])
            p = float(tokens[2])
            q = float(tokens[3])
            cumulative = True if tokens[4] == "True" else False

            self.configs[dsname] = (dim, p, q, cumulative)

    def get_embedding(self, dataset: str) -> Embedding:
        if not dataset in self.configs:
            print("WARNING: dataset not in ctdne configuration")
            return None  # type: ignore

        conf = self.configs[dataset]

        es, time_resolution = load_event_sequence(dataset)  # type: ignore

        #dg = DynamicGraph()
        #dg.init_from_event_sequence(
        #    es, time_resolution=time_resolution, directed=False, cumulative=conf[3]
        #)
        
        ctdne = Ctdne(dim=conf[0], p=conf[1], q=conf[2])
        agg_dg = DynamicAggregateGraph(name=dataset, event_sequence=es, directed=False)

        ctdne.make_embedding(agg_dg)

        return ctdne, agg_dg  # type: ignore



class DynDeepHubWalk_Factory(Factory):
    def __init__(self, deep_hub_conf, datapath=None):
        super().__init__(datapath)

        conf_file = open(deep_hub_conf, "r")
        conf_file.readline()

        self.configs = dict()

        for line in conf_file:
            tokens = line.split(",")
            dataset_name = tokens[0]
            dim = int(tokens[1])
            inverse = True if tokens[2] == "True" else False
            log_scaling = True if tokens[3] == "True" else False
            p = float(tokens[4])
            u = float(tokens[5])
            cumulative = True if tokens[6] == "True" else False

            self.configs[dataset_name] = (dim, inverse, log_scaling, p, u, cumulative)
        
    def get_embedding(self, dataset: str) -> Embedding:
        if not dataset in self.configs:
            print("WARNING: dataset not in DeepHub configuration!")
            return None  

        conf = self.configs[dataset]

        deep_hub = DynDeepHubWalk(dim=conf[0],
            inverse=conf[1],
            log_scaling=conf[2],
            p=conf[3],
            u=conf[4],
        )

        es, time_resolution = load_event_sequence(dataset)
        dg = DynamicGraph()
        dg.init_from_event_sequence(
            es, time_resolution=time_resolution, directed=False, cumulative=conf[5]
        )

        deep_hub.make_embedding(dg)
        
        return deep_hub, dg



dn2v_factory = None
stw_factory = None
ctdne_factory = None
deep_hub_factory = None


def get_dn2v_factory(dn2vconf, datapath=None):
    global dn2v_factory

    if dn2v_factory == None:
        dn2v_factory = DN2V_Factory(dn2vconf, datapath=datapath)

    return dn2v_factory

def get_stwalk_factory(stwalkconf, datapath=None):
    global stw_factory

    if stw_factory == None:
        stw_factory = STWalk_Factory(stwalkconf, datapath=datapath)

    return stw_factory

def get_ctdne_factory(ctdneconf, datapath=None):
    global ctdne_factory

    if ctdne_factory == None:
        ctdne_factory = CTDNE_Factory(ctdneconf, datapath=datapath)

    return ctdne_factory


def get_deep_hub_factory(deep_hub_conf, datapath=None):
    global deep_hub_factory

    if deep_hub_factory == None:
        deep_hub_factory = DynDeepHubWalk_Factory(deep_hub_conf, datapath=datapath)
    
    return deep_hub_factory