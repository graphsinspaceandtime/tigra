import sys
import os

sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True

from tigra.embeddings.sota.factory import get_deep_hub_factory
from data.bank import Databank


deep_hub_fact = get_deep_hub_factory("../deep_hub.conf")

datasets = [
    "radoslaw-email",
    "fb-forum",
    "fb-messages",
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    "ia-primary-school-proximity-attr",
    "CollegeMsg",
    "email-Eu-core-temporal",
]


for dataset in datasets:
    print("DATASET", dataset)
    deep_hub, dg = deep_hub_fact.get_embedding(dataset)
    deep_hub.save_embedding(dataset)
