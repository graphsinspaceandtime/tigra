import sys

sys.path.append("../..")

from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.dynae_wrapper import DynAE
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month")

method = DynAE(
    dim=10,
    n_units=[500, 300],
    look_back=5,
    beta=5,
    nu1=1e-6,
    nu2=1e-6,
    batch_size=200,
    epochs=10,
    learning_rate=0.001,
    weight_decay=1e-4,
)
method.make_embedding(dsmg)
print(method.embeddings[0])
