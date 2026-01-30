import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

import sys
import warnings
import torch
import numpy as np
import random


from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.utils import resample


from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


sys.path.append("..") 
sys.dont_write_bytecode = True


class BaseAdapter:

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_binarycl(self, X_train, y_train, X_test, y_test):

        self.fit(X_train, y_train)
        
        preds = self.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        
        try:
            tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0

        pos_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pos_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        neg_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        neg_rec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return f1, acc, prec, rec, pos_prec, pos_rec, neg_prec, neg_rec


class TabPFNAdapter(BaseAdapter):
    def __init__(self, use_gpu=True):
        self.model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, ignore_pretraining_limits=True
        )
        self.name = "TabPFN"
        if use_gpu and torch.cuda.is_available():
            self.model.device = 'cuda'
        else:
            self.model.device = 'cpu'

    def fit(self, X, y):
        MAX_SAMPLES = 800 
        if len(X) > MAX_SAMPLES:
            X, y = resample(X, y, n_samples=MAX_SAMPLES, random_state=42, stratify=y)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Running on CPU")
            return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True): return {}


class XGBAdapter(BaseAdapter):
    def __init__(self):
        self.name = "XGBoost"
        self.model = XGBClassifier(eval_metric='logloss', n_estimators=100, n_jobs=1)

    def fit(self, X, y): 
        return self.model.fit(X, y)

    def predict(self, X): 
        return self.model.predict(X)

    def get_params(self, deep=True): 
        return self.model.get_params(deep)


class LGBMAdapter(BaseAdapter):
    def __init__(self):
        self.name = "LightGBM"
        self.model = LGBMClassifier(n_estimators=100, n_jobs=1, verbose=-1)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
            return self.model.fit(X, y)

    def predict(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
            return self.model.predict(X)

    def get_params(self, deep=True): 
        return self.model.get_params(deep)


class CatBoostAdapter(BaseAdapter):
    def __init__(self):
        self.name = "CatBoost"
        self.model = CatBoostClassifier(iterations=100, verbose=0, thread_count=1)

    def fit(self, X, y): 
        return self.model.fit(X, y)

    def predict(self, X): 
        return self.model.predict(X)

    def get_params(self, deep=True): 
        return self.model.get_params(deep)



from tigra.evaluation.diffusion import DiffusionPrediction

class CustomDiffusionPrediction(DiffusionPrediction):
   
    def __init__(self, embedding, dg, diffusion_process, classifiers_list, **kwargs):
        
        super().__init__(embedding, dg, diffusion_process, **kwargs)
        
    
        self.ALL_MODELS = []
        for clf in classifiers_list:
            self.ALL_MODELS.append((clf, clf.name))
        
        self.RF_MODELS = []
        self.KNN_MODELS = []
        self.NB_MODELS = []
        self.SVM_MODELS = []
        self.LR_MODELS = []


if __name__ == "__main__":
    from tigra.core.dgraph import DynamicGraph
    from tigra.core.loaders import load_event_sequence
    from tigra.embeddings.embedding import FSEmbedding
    from tigra.evaluation.diffusion import SI

    print("--- 1. Loading Data ---")
    es, resol = load_event_sequence("radoslaw-email")
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)

    print("--- 2. Loading Embeddings ---")
    embedding_path = ""
    emb = FSEmbedding(embedding_path)

    my_classifiers = [
        TabPFNAdapter(use_gpu=True),
        XGBAdapter(),
        LGBMAdapter(),
        CatBoostAdapter()
    ]

    print("\n\n=== DIFFUSION PREDICTION BENCHMARK ===")
    

    si = SI(verbose=False)


    dp = CustomDiffusionPrediction(
        emb, 
        dg, 
        si, 
        classifiers_list=my_classifiers,  
        negative_sampling=True, 
        verbose=True
    )
    
    print("Initializing Evaluation Data...")
    dp.init_evaluation()
    
    print("\nRunning Evaluation...")

    f1, p, r, cl_name = dp.evaluate(model_type="ALL")

    print("\n\n>>> WINNER:")
    print(f"Model: {cl_name}")
    print(f"F1:    {round(f1, 4)}")
    print(f"Prec:  {round(p, 4)}")
    print(f"Rec:   {round(r, 4)}")