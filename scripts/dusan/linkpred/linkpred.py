from tigra.embeddings.embedding import FSEmbedding
from tigra.data.loaders import IaContactsHypertext2009
from tigra.core.dgraph import DynamicGraph
from tigra.evaluation.linkpred import distance_measure_factory, UnsupervisedLinkPrediction, Predicter

embed = FSEmbedding(folder_name="../make_embed/embs_ae_unsup/ia_contacts_hypertext_2009")
data = IaContactsHypertext2009("../../../data")

es = data.get_event_sequence(shrink=True)

dg = DynamicGraph()
dg.init_from_event_sequence(
  event_sequence=es,
  time_resolution=data.resolution,
  directed=False,
  cumulative=data.cumulative,
)

for distance_measure in distance_measure_factory():
  print("distance" + distance_measure.name)
  linkpred = UnsupervisedLinkPrediction(distance_measure)
  predicter = Predicter(linkpred, embed, dg)
  predicter.predict()
