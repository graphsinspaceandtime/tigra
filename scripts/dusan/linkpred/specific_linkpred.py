import os
import sys

from tigra.embeddings.embedding import FSEmbedding
from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicGraph

from tigra.evaluation.linkpred import (
  UnsupervisedLinkPrediction,
  SupervisedLinkPrediction,
  Predicter,
  distance_measure_factory,
  combine_pair_operation_factory,
)

#from tigra.evaluation.linkpred_classification.skl import skl_classifiers_factory
from tigra.evaluation.clbase.skl import skl_classifiers_factory

embeddings_path = "../../../embeddings"
datasets_path = "../../../data"

pred_type = sys.argv[1].strip()
method = sys.argv[2].strip()
dataset_id = sys.argv[3].strip()

method_path = os.path.join(embeddings_path, method)
embedding_path = os.path.join(method_path, dataset_id)

embedding = FSEmbedding(embedding_path)
print(embedding.embeddings)

Data = get_dataset_by_id(dataset_id)
data = Data(base_path=datasets_path)

es = data.get_event_sequence(shrink=True)
dg = DynamicGraph()
dg.init_from_event_sequence(
  event_sequence=es,
  time_resolution=data.resolution,
  directed=False,
  cumulative=data.cumulative,
)

if pred_type == "unsupervised":
  for distance_measure in distance_measure_factory():
    linkpred = UnsupervisedLinkPrediction(distance_measure)
    predicter = Predicter(linkpred, embedding, dg)
    try:
      predicter.predict()
      report_file = (
        f"results/unsup~{method}~{dataset_id}~{distance_measure.name}.json"
      )
      predicter.save_report(report_file)
      print("saved " + method + " " + dataset_id + " " + distance_measure.name)
    except Exception as e:
      print(e)
elif pred_type == "supervised":
  for classifier in skl_classifiers_factory():
    for combine_pair_operation in combine_pair_operation_factory():
      linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
      predicter = Predicter(linkpred, embedding, dg)
      try:
        predicter.predict()
        report_file = f"results/sup~{method}~{dataset_id}~{classifier.name}~{combine_pair_operation.name}.json"
        predicter.save_report(report_file)
        print("saved " + method + " " + dataset_id + " " + classifier.name + " " + combine_pair_operation.name)
      except Exception as e:
        print(e)

