# primary (graph reconstruction) evaluation of dynamic graph embedding methods based on temporal random walks

import sys

sys.path.append("..")
sys.dont_write_bytecode = True


from data.bank import Databank
from tigra.core.loaders import load_event_sequence

from tigra.core.dgraph import TemporalAggregateGraph
from tigra.embeddings.rndw.trndwbase import DeepTemporalWalk
from tigra.embeddings.rndw.trndw import DeepLIDTemporalWalk, DeepHubTemporalWalk
from tigra.core.greval import GraphReconstructionEval

databank = Databank()
datasets = databank.get_keys()

def eval_on_dataset(index):
    dataset_name = datasets[index]
    print("Evaluating on", dataset_name)
    es, resol = load_event_sequence(dataset_name)
    tag = TemporalAggregateGraph(dataset_name + "-tag", es)

if __name__ == "__main__":
    eval_on_dataset(0)
