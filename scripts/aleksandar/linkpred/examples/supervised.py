from tigra.embeddings.embedding import FSEmbedding
from tigra.data.loaders import (
    RadoslawEmail,
    FbForum,
    FbMessages,
    IaContactsHypertext2009,
    IaEnronEmployees,
    IaHospitalWardProximityAttr,
    IaPrimarySchoolProximityAttr,
    CollegeMsg,
    EmailEuCoreTemporal
  )
from tigra.core.dgraph import DynamicGraph
from tigra.evaluation.linkpred import combine_pair_operation_factory, SupervisedLinkPrediction, Predicter

#from tigra.evaluation.linkpred_classification.skl import skl_classifiers_factory
from tigra.evaluation.clbase.skl import skl_classifiers_factory

embed = FSEmbedding(folder_name="embeddings/dn2v/fb_forum")
data = FbForum("../../../../data")

es = data.get_event_sequence(shrink=True)

dg = DynamicGraph()
dg.init_from_event_sequence(
  event_sequence=es,
  time_resolution=data.resolution,
  directed=False,
  cumulative=data.cumulative,
)

for classifier in skl_classifiers_factory():
  for combine_pair_operation in combine_pair_operation_factory():
    linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
    predicter = Predicter(linkpred, embed, dg)
    predicter.predict(print_results=True)
