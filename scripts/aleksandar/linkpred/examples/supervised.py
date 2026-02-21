from tigra.embeddings.embedding import FSEmbedding
from tigra.data.loaders import (
    RadoslawEmail,
    FbForum,
    FbMessages,
    IaEnronEmployees,
    IaHospitalWardProximityAttr,
    CollegeMsg,
    EmailEuCoreTemporal,
)
from tigra.core.dgraph import DynamicGraph
from tigra.evaluation.linkpred import combine_pair_operation_factory, SupervisedLinkPrediction, Predicter

from tigra.evaluation.clbase.skl import skl_classifiers_factory
from tigra.evaluation.clbase.skl import rf_tuning_factory, knn_tuning_factory, svm_tuning_factory, lr_tuning_factory

"""
RF: classifier.model.n_estimators

KNN: classifier.model.n_neighbors

SVM: classifier.model.kernel

Logistic Regression: classifier.model.penalty
"""

embed = FSEmbedding(folder_name="embeddings/deep_hub/email-Eu-core-temporal")
data = EmailEuCoreTemporal("../../../../data")

es = data.get_event_sequence(shrink=True)

dg = DynamicGraph()
dg.init_from_event_sequence(
    event_sequence=es,
    time_resolution=data.resolution,
    directed=False,
    cumulative=data.cumulative,
)

"""
for classifier in skl_classifiers_factory():
    for combine_pair_operation in combine_pair_operation_factory():
        linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
        predicter = Predicter(linkpred, embed, dg)
        predicter.predict(print_results=True)
"""

# Supervised LP tuning
import csv
import os
import numpy as np

output_file = "tuning_results_detailed.csv"

csv_headers = [
    "Classifier", "Param_Name", "Param_Value", "Combiner", 
    "Snapshot_Index", "F1", "Accuracy", "Precision", "Recall", 
    "Pos_Prec", "Pos_Rec", "Neg_Prec", "Neg_Rec"
]


with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

print(f"Results will be saved to: {output_file}")


best_global_f1 = -1.0
best_global_config = {}


factories_to_test = [
    rf_tuning_factory,
    knn_tuning_factory,
    svm_tuning_factory,
    lr_tuning_factory
]

print("Starting Global Grid Search...")

for factory_func in factories_to_test:
    classifiers = factory_func()
    
    for classifier in classifiers:
        for combine_pair_operation in combine_pair_operation_factory():
            
            linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
            predicter = Predicter(linkpred, embed, dg)

            param_name = "None"
            param_val = "None"
            
            if classifier.name == "RF":
                param_name = "n_estimators"
                param_val = classifier.model.n_estimators
            elif classifier.name == "KNN":
                param_name = "n_neighbors"
                param_val = classifier.model.n_neighbors
            elif classifier.name == "SVM":
                param_name = "kernel"
                param_val = classifier.model.kernel
            elif classifier.name == "LOGREG":
                param_name = "penalty"
                param_val = classifier.model.penalty

            print(f"Testing {classifier.name} [{param_name}={param_val}] with {combine_pair_operation.name}...")


            predicter.predict(print_results=False) 

            with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                for idx, res in enumerate(predicter.results, start=1):
                    row = [
                        classifier.name,
                        param_name,
                        param_val,
                        combine_pair_operation.name,
                        idx, # Snapshot number
                        res.get('f1', 0),
                        res.get('accuracy', 0),
                        res.get('precision', 0),
                        res.get('recall', 0),
                        res.get('pos_prec', 0),
                        res.get('pos_rec', 0),
                        res.get('neg_prec', 0),
                        res.get('neg_rec', 0)
                    ]
                    writer.writerow(row)


            run_f1_scores = [res.get('f1', 0) for res in predicter.results]
            if len(run_f1_scores) > 0:
                avg_f1 = np.mean(run_f1_scores)
                if avg_f1 > best_global_f1:
                    best_global_f1 = avg_f1
                    best_global_config = {
                        "classifier": classifier.name,
                        "param": f"{param_name}={param_val}",
                        "combiner": combine_pair_operation.name,
                        "avg_f1": best_global_f1
                    }

print("\n" + "#"*60)
print(" GLOBAL WINNER ")
print("#"*60)
if best_global_config:
    print(f"Best Average F1: {best_global_config['avg_f1']:.5f}")
    print(f"Model:           {best_global_config['classifier']}")
    print(f"Configuration:   {best_global_config['param']}")
    print(f"Combiner:        {best_global_config['combiner']}")
    print(f"Full details saved to {output_file}")
else:
    print("No results found.")
print("#"*60)