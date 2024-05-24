import sys
import os
import time

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from data.bank import Databank

from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.hypertuner import Dynnode2vec_Tuner

databank = Databank()
es = databank.get_data("radoslaw-email")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False)
dg.info()

dn2v = Dynnode2vec(10)
dn2v.make_embedding(dg)

org_graphs = dg.get_graph_sequence()

last_embedding = dn2v.embeddings[-1]
last_graph = org_graphs[-1]
print("Original graph", last_graph.num_nodes(), last_graph.num_edges())
print("#emb-vectors", len(last_embedding.embedding.keys()))

gre = GraphReconstructionEval(last_graph, last_embedding)
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())


print("TESTING TUNER")
start_time = time.time()
tuner = Dynnode2vec_Tuner(last_graph, dg, [0.5, 1.0, 2.0, 4.0], [0.5, 1.0, 2.0, 4.0], num_runs=10, parallel_processes=16, walk_length=64, n_walks_per_node=10)
# tuner = Dynnode2vec_Tuner(last_graph, dg, [0.5, 1.0], [0.5, 1.0], num_runs=5)
tuner.tune()
tuner.print_report()
end_time = time.time()
elapsed_time = end_time - start_time
print("TIME [sec]", elapsed_time)
