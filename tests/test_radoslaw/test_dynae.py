from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.ae_wrapper import DynamicAutoEncoder
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month")

dynae = DynamicAutoEncoder(10, 2)
dynae.make_embedding(dsmg)
print(dynae.embeddings[0])
