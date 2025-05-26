import sys
import os

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from tigra.embeddings.sota.factory import get_deep_hub_factory
from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

deep_hub_fact = get_deep_hub_factory("../deep_hub.conf")


for dataset in datasets:
    print("DATASET", dataset)
    deep_hub, dg = deep_hub_fact.get_embedding(dataset)
    deep_hub.save_embedding(dataset)