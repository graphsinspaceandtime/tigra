import sys
import os

workers = int(os.environ.get("NUM_CPUS", 4))

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicAggregateGraph
from tigra.embeddings.sota.ctdne_wrapper import Ctdne
import data.bank as databank

from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.hypertuner import CTDNE_Tuner

es = databank.get_data("radoslaw-email")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

dag = DynamicAggregateGraph("radoslaw", es, directed=False)
print("DIRECTED = ", dag.get_aggregate_graph().is_directed())

ctdne = Ctdne(10, workers=workers)
ctdne.make_embedding(dag)
#print(ctdne.is_singleton_embedding())
#print(ctdne.embeddings[0])

print("\nGRAPH RECONSTRUCTION")
gre = GraphReconstructionEval(dag.get_aggregate_graph(), ctdne.embeddings[0])
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())

print("TESTING TUNER")
tuner = CTDNE_Tuner(dag.get_aggregate_graph(), dag, [0.5, 1.0, 2.0, 4.0], [0.5, 1.0, 2.0, 4.0], num_runs=5)
tuner.tune()
tuner.print_report()
