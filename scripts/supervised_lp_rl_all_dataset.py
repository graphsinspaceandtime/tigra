import sys
import os
import csv
import random
import numpy as np
import networkx as nx


sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph


try:
    from tigra.data.bank import Databank
except ImportError:
    try:
        from data.bank import Databank
    except ImportError:
        from tigra.data.loaders import Databank

from tigra.evaluation.linkpred import combine_pair_operation_factory, SupervisedLinkPrediction, Predicter


from tigra.evaluation.clbase.skl import (
    skl_classifiers_factory,
    rf_tuning_factory,
    knn_tuning_factory,
    svm_tuning_factory,
    lr_tuning_factory
)

from tigra.embeddings.rndw.rlwalk import DynRLWalk


DATASET_CONFIGS = {
    "radoslaw-email": {
        "time_res": "month",
        "params": {'dim': 25, 'eps': 0.3, 'alpha': 0.1, 'bias': False}
    },
    "ia-hospital-ward-proximity-attr": {
        "time_res": "day",
        "params": {'dim': 100, 'eps': 0.1, 'alpha': 0.1, 'bias': False}
    },
    "ia-contacts_hypertext2009": {
        "time_res": "day", 
        "params": {'dim': 10, 'eps': 0.2, 'alpha': 0.1, 'bias': True}
    },
    "ia-enron-employees": {
        "time_res": "3-months", 
        "params": {'dim': 50, 'eps': 0.1, 'alpha': 0.1, 'bias': False}
    },
    "ia-primary-school-proximity-attr": {
        "time_res": "day",
        "params": {'dim': 10, 'eps': 0.3, 'alpha': 0.1, 'bias': False}
    },
    "fb-forum": {
        "time_res": "month",
        "params": {'dim': 50, 'eps': 0.3, 'alpha': 0.1, 'bias': False}
    },
    "email-Eu-core-temporal": {
        "time_res": "month",
        "params": {'dim': 200, 'eps': 0.2, 'alpha': 0.1, 'bias': True}
    },
    "CollegeMsg": {
        "time_res": "month",
        "params": {'dim': 100, 'eps': 0.2, 'alpha': 0.1, 'bias': True}
    },
    "fb-messages": {
        "time_res": "month",
        "params": {'dim': 200, 'eps': 0.3, 'alpha': 0.1, 'bias': False}
    }
}


class RamEmbedding:
    """Wraps a list of embeddings to mimic FSEmbedding interface for Predicter."""
    def __init__(self, embedding_list):
        self.embeddings = embedding_list
    
    def num_embeddings(self):
        return len(self.embeddings)
        
    def get_embedding(self, index):
        if 0 <= index < len(self.embeddings):
            return self.embeddings[index]
        return None
    
    def __getitem__(self, index):
        return self.embeddings[index]
    
    def __len__(self):
        return len(self.embeddings)


if __name__ == "__main__":
    
    databank = Databank()
    

    for dataset_name, config in DATASET_CONFIGS.items():

        print(f"PROCESSING: {dataset_name}")
        print(f"Config:     {config}")

        
        output_file = f"supervised_results_{dataset_name}.csv"
        
        try:

            es = databank.get_data(dataset_name)
            dg = DynamicGraph()
            dg.init_from_event_sequence(es, time_resolution=config["time_res"], directed=False)
            
            num_graphs = len(dg.get_graph_sequence())
            if num_graphs < 3:
                print(f"[SKIP] Not enough snapshots ({num_graphs}). Need >= 3.")
                continue


            print(f"[1/3] Generating RL Embeddings...")
            rl_model = DynRLWalk(
                dim=config['params']['dim'], 
                epsilon=config['params']['eps'], 
                alpha=config['params']['alpha'], 
                reward_bias=config['params']['bias'], 
                num_walks=10, 
                walk_length=40, 
                num_workers=1
            )
            raw_embeddings = rl_model.construct(dg)
            embed_wrapper = RamEmbedding(raw_embeddings)
            

            csv_headers = [
                "Classifier", "Param_Name", "Param_Value", "Combiner", 
                "Snapshot_Index", "F1", "Accuracy", "Precision", "Recall", 
                "Pos_Prec", "Pos_Rec", "Neg_Prec", "Neg_Rec"
            ]
            
            with open(output_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)

            print(f"[2/3] Running Grid Search -> {output_file}")
            
            best_global_f1 = -1.0
            best_global_config = {}


            factories_to_test = [svm_tuning_factory]

            for factory_func in factories_to_test:
                classifiers = factory_func()
                
                for classifier in classifiers:
                    for combine_pair_operation in combine_pair_operation_factory():
                        

                        linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
                        predicter = Predicter(linkpred, embed_wrapper, dg)


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

                        print(f"      Testing {classifier.name} [{param_name}={param_val}] + {combine_pair_operation.name}...")

                        try:
                            predicter.predict(print_results=False) 

                            with open(output_file, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                
                                for idx, res in enumerate(predicter.results, start=1):
                                    row = [
                                        classifier.name, param_name, param_val, combine_pair_operation.name,
                                        idx, 
                                        res.get('f1', 0), res.get('accuracy', 0), res.get('precision', 0), res.get('recall', 0),
                                        res.get('pos_prec', 0), res.get('pos_rec', 0), res.get('neg_prec', 0), res.get('neg_rec', 0)
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
                        
                        except Exception as e:
                            pass

            print(f" WINNER FOR {dataset_name} ")
            if best_global_config:
                print(f"Best Average F1: {best_global_config.get('avg_f1'):.5f}")
                print(f"Model:           {best_global_config.get('classifier')}")
                print(f"Configuration:   {best_global_config.get('param')}")
                print(f"Combiner:        {best_global_config.get('combiner')}")
                print(f"Full details saved to {output_file}")
            else:
                print("No results found.")

        except Exception as e:
            import traceback
            print(f"[ERROR] Fatal error on {dataset_name}: {e}")
            traceback.print_exc()
            continue