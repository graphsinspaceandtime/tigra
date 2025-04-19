from abc import ABC, abstractmethod
from typing import Union, Tuple
import json

from tigra.data.loaders import get_dataset_by_id

from tigra.embeddings.sota.dynae_wrapper import DynAE

from tigra.embeddings.embedding import Embedding
from tigra.core.dgraph import DynamicSparseMatrixGraph, DynamicGraph

class Factory(ABC):
    datapath: Union[str, None]
    common_configs: dict = {}
    dataset_configs: dict = {}


    def __init__(self, datapath=None):
        self.datapath = datapath

    @abstractmethod
    def get_embedding(self, dataset_id: str, has_cuda=False) -> Union[Tuple[Embedding, DynamicGraph], None]:
        pass

class DynAE_Factory(Factory):
    def __init__(self, dynae_config: str, datapath=None):
        super().__init__(datapath)
        configs = json.load(open(dynae_config, "r", encoding="utf-8"))
        self.common_configs = configs["common_configs"]
        self.dataset_configs = configs["dataset_configs"]

    def get_embedding(self, dataset_id, has_cuda=False):
        if not dataset_id in self.dataset_configs:
            print("WARNING: Selected dataset not in factory configuration")
            return None
        
        if self.dataset_configs[dataset_id]["skip"]:
            print("WARNING: Selected dataset configuration is skipped")
            return None

        config = self.dataset_configs[dataset_id]

        dynae = DynAE(
            dim=config["dim"],
            n_units=config["n_units"],
            look_back=config["look_back"],
            has_cuda=has_cuda,
            **self.common_configs
        )

        Data = get_dataset_by_id(dataset_id)
        data = Data(self.datapath)

        es = data.get_event_sequence()

        dsmg = DynamicSparseMatrixGraph(
            name=dataset_id,
            event_sequence=es,
            time_resolution=data.resolution,
            directed=False,
            cumulative=data.cumulative,
        )

        dg = DynamicGraph()
        dg.init_from_event_sequence(
            event_sequence=es,
            time_resolution=data.resolution,
            directed=False,
            cumulative=data.cumulative,
        )

        dynae.make_embedding(dsmg)

        return dynae, dg

dynae_factory = None

def get_dynae_factory(dynae_config, datapath=None):
    global dynae_factory

    if dynae_factory == None:
        dynae_factory = DynAE_Factory(dynae_config, datapath)

    return dynae_factory
