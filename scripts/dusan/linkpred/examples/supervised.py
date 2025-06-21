from tigra.embeddings.embedding import FSEmbedding
from tigra.data.loaders import RadoslawEmail
from tigra.core.dgraph import DynamicGraph
from tigra.evaluation.linkpred import distance_measure_factory, UnsupervisedLinkPrediction, Predicter

embed = FSEmbedding(folder_name="../../../../embeddings/dn2v/radoslaw_email")
data = RadoslawEmail("../../../../data")

es = data.get_event_sequence(shrink=True)

dg = DynamicGraph()
dg.init_from_event_sequence(
  event_sequence=es,
  time_resolution=data.resolution,
  directed=False,
  cumulative=data.cumulative,
)

for distance_measure in distance_measure_factory():
  linkpred = UnsupervisedLinkPrediction(distance_measure)
  predicter = Predicter(linkpred, embed, dg)
  predicter.predict(print_results=True)
