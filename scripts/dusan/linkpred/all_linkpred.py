import os
from tqdm import tqdm

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
from tigra.evaluation.linkpred_classification.skl import skl_classifiers_factory

embeddings_path = "../../../embeddings"
datasets_path = "../../../data"

methods = [
    method
    for method in os.listdir(embeddings_path)
    if os.path.isdir(os.path.join(embeddings_path, method))
]

for method in tqdm(methods, desc="Methods"):
    method_path = os.path.join(embeddings_path, method)
    dataset_ids = [
        dataset_id
        for dataset_id in os.listdir(method_path)
        if os.path.isdir(os.path.join(method_path, dataset_id))
    ]
    for dataset_id in tqdm(dataset_ids, desc="Datasets", leave=False):
        embedding_path = os.path.join(method_path, dataset_id)

        embedding = FSEmbedding(embedding_path)

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

        # unsupervised link prediction
        for distance_measure in tqdm(distance_measure_factory(), desc="Unsupervised", leave=False):
            # print("=== Unsupervised Link Prediction ===")
            # print(f"method: {method}")
            # print(f"dataset: {dataset_id}")
            # print(f"distance: {distance_measure.name}")
            # print()

            linkpred = UnsupervisedLinkPrediction(distance_measure)
            predicter = Predicter(linkpred, embedding, dg)
            try:
                predicter.predict()

                report_file = (
                    f"results/unsup~{method}~{dataset_id}~{distance_measure.name}.json"
                )
                predicter.save_report(report_file)

                # print(f"report: {report_file}")
            except Exception as e:
                pass
                # print(e)

            # print("====================================")

        # supervised link prediction
        for classifier in tqdm(skl_classifiers_factory(), desc="Supervised", leave=False):
            for combine_pair_operation in combine_pair_operation_factory():
                # print("=== Unsupervised Link Prediction ===")
                # print(f"method: {method}")
                # print(f"dataset: {dataset_id}")
                # print(f"classifier: {classifier.name}")
                # print(f"combiner: {combine_pair_operation.name}")
                # print()

                linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
                predicter = Predicter(linkpred, embedding, dg)
                try:
                    predicter.predict()

                    report_file = f"results/sup~{method}~{dataset_id}~{classifier.name}~{combine_pair_operation.name}.json"
                    predicter.save_report(report_file)

                    # print(f"report: {report_file}")
                except Exception as e:
                    pass
                    # print(e)

                # print("====================================")
