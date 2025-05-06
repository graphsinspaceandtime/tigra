# For Mars
# hackathon 22.2.2025

import sys
import os

sys.path.append("../..")

from tigra.generators.dsbm import DPPGenerator
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.rndw.rndwbase import DynN2V
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.embedding import FSEmbedding

from data.bank import Databank


# step 1 -- create a databank instance
print("\n\n###### step 1 -- create a databank instance ####\n")

db = Databank("../../data")

# choose a labeled dataset
dataset_name = "ia-hospital-ward-proximity-attr"


# step 2 -- load event sequence and metadata
print("\n\n###### step 2 -- load event sequence and metadata ######\n")

es = db.get_data(dataset_name)
metadata = db.get_metadata(dataset_name)


# step 3 -- create a temporal graph from the loaded event sequence
print("\n\n###### step 3 -- create a temporal graph from the loaded event sequence ######\n")

dg = DynamicGraph()
dg.init_from_event_sequence(
    es,
    time_resolution=metadata["resolution"],
    directed=False,
    cumulative=metadata["cumulative"],
)
dg.info()


# step 4 -- create embedding
print("\n\n###### step 4 -- create embedding #############\n")

dn2v = DynN2V(200, 0.5, 4)
dn2v.make_embedding(dg)


# step 5 -- save embedding
print("\n\n###### step 5 -- save embedding ###############\n")

folder_name = dataset_name
dn2v.save_embedding(folder_name)


# step 6 -- eval singleton embeddings and save labels
print("\n\n###### step 6 -- eval singleton embeddings and save labels ######\n")

graph_sequence = dg.get_graph_sequence()
for i in range(len(graph_sequence)):
    print("SNAPSHOT", i)
    emb = dn2v.embeddings[i]
    graph = graph_sequence[i]

    gre = GraphReconstructionEval(graph, emb)

    print("Precision", gre.get_avg_precision())
    print("Recall", gre.get_avg_recall())
    print("F1", gre.getF1())

    fn = str(i) + ".labels"
    path = os.path.join(folder_name, fn)
    graph.export_labels(path)


# step 7 -- load embedding with labels
print("\n\n###### step 7 -- load embedding with labels ######\n")

emb = FSEmbedding(folder_name)
print("#embeddings", len(emb.embeddings))
for e in emb:
    nodes = e.get_node_ids()
    node_0 = nodes[0]
    emb_0 = e[node_0]
    label_0 = e.get_label(node_0)
    print(node_0, "label", label_0, "emb dimension", len(emb_0))
    #print(emb_0)
    print()
