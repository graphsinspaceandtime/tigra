# For Mars
# hackaton 22.2.2025

import sys
import os

sys.path.append("..")

from tigra.generators.dsbm import DPPGenerator
from tigra.core.loaders import load_snap_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.rndw.rndwbase import DynN2V
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.embedding import FSEmbedding

from data.bank import Databank

# step 1 -- create a databank instance
db = Databank("../data")

# choose a labeled dataset
dataset_name = "ia-primary-school-proximity-attr"

# step 2 -- load event sequence and metadata
es = db.get_data(dataset_name)
metadata = db.get_metadata(dataset_name)

# step 3 -- create temporal graph form loaded event sequence
dg = DynamicGraph()
dg.init_from_event_sequence(
    es,
    time_resolution=metadata["resolution"],
    directed=False,
    cumulative=metadata["cumulative"],
)
dg.info()

# step 4 -- create embedding
dn2v = DynN2V(200, 0.5, 4)
dn2v.make_embedding(dg)

# step 5 -- save embedding
folder_name = dataset_name
dn2v.save_embedding(folder_name)

# step 6 -- eval singleton embeddings and save labels
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
emb = FSEmbedding(folder_name)
print("#embeddings", len(emb.embeddings))
for e in emb:
    nodes = e.get_node_ids()
    node_0 = nodes[0]
    emb_0 = e[node_0]
    label_0 = e.get_label(node_0)
    print(node_0, "label", label_0, "emb dimension", len(emb_0))
    print(emb_0)
    print()
