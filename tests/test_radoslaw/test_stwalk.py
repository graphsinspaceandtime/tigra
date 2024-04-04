import sys
import os

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.sota.stwalk import STWalk
import data.bank as databank

from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.hypertuner import STWalk_Tuner

es = databank.get_data("radoslaw-email")
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False)
dg.info()

org_graphs = dg.get_graph_sequence()

ts = len(dg.get_nx_graph_sequence())
st = STWalk(
    number_restart=40,
    walk_length=10,
    dimension=5,
    time_step=ts,
    time_window_size=1,
    workers=4,
    vocab_window_size=5,
)
st.make_embedding(dg)

last_embedding = st.embeddings[-1]
last_graph = org_graphs[-1]
print("Original graph", last_graph.num_nodes(), last_graph.num_edges())
print("#emb-vectors", len(last_embedding.embedding.keys()))

gre = GraphReconstructionEval(last_graph, last_embedding)
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())

print("TESTING TUNER")
tuner = STWalk_Tuner(last_graph, dg, [1, 3, 5], num_runs=5, n_walks_per_node=40)
tuner.tune()
tuner.print_report()
