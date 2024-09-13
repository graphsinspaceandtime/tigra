from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.dyngem_wrapper import DynGEM
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month")

method = DynGEM(
    dim=10,
    n_units=[500, 300],
    alpha=5,
    beta=5,
    nu1=1e-6,
    nu2=1e-6,
    batch_size=200,
    epochs=10,
    learning_rate=0.001,
)
method.make_embedding(dsmg)
print(method.embeddings[0])
