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

# step 1 -- generate and save the event sequence of a synthetic graph with labels
dppg = DPPGenerator([500, 250, 150, 100], 0.03, 0.001)
dppg.generate_and_save("dpp_1000_4.events")

# step 2 -- load event sequence
es = load_snap_dataset("dpp_1000_4.events")

# step 3 -- create temporal graph from loaded event sequence
dg = DynamicGraph()
dg.init_from_event_sequence(
    es, time_resolution="month", directed=False, cumulative=True
)
dg.info()

# step 3 -- create embedding
dn2v = DynN2V(100, 0.5, 4)
dn2v.make_embedding(dg)

# step 4 -- save embedding
folder_name = "dpp_labeled"
dn2v.save_embedding(folder_name)

# step 5 -- eval singleton embeddings and save labels
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

# step 6 -- load embedding with labels
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
