# applicative evaluation of dynamic graph embedding methods based on temporal random walks

import sys

sys.path.append("..")
sys.dont_write_bytecode = True


from data.bank import Databank
from tigra.core.loaders import load_event_sequence

from tigra.evaluation.trw_eval import SLPEval, ULPEval

import random 
import numpy as np

databank = Databank()
datasets = databank.get_keys()
print("DATASETS")
print(datasets)
print()

def test_slp(index):
    dataset_name = datasets[index]
    print("Supervised link prediction evaluation on ", dataset_name)
    es, _ = load_event_sequence(dataset_name)

    slpe = SLPEval(index, es)
    slpe.eval()

def test_ulp(index):
    dataset_name = datasets[index]
    print("Unsupervised linke prediction evaluation on ", dataset_name)
    es, _ = load_event_sequence(dataset_name)

    ulpe = ULPEval(index, es)
    ulpe.eval()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    test_ulp(0)