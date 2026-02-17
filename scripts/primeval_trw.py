# primary (graph reconstruction) evaluation of dynamic graph embedding methods based on temporal random walks

import sys

sys.path.append("..")
sys.dont_write_bytecode = True


from data.bank import Databank
from tigra.core.loaders import load_event_sequence

from tigra.core.dgraph import TemporalAggregateGraph
from tigra.embeddings.hypertuner import DeepLIDTemporalWalk_Tuner
from tigra.embeddings.hypertuner import DeepHUBTemporalWalk_Tuner
from tigra.embeddings.hypertuner import DeepTemporalWalk_Tuner

databank = Databank()
datasets = databank.get_keys()

def eval_on_dataset(index):
    dataset_name = datasets[index]
    print("Evaluating on", dataset_name)
    es, resol = load_event_sequence(dataset_name)
    tag = TemporalAggregateGraph(dataset_name + "-tag", es)

    print("DEEP-TEMPORAL-WALK")
    tuner = DeepTemporalWalk_Tuner(tag, tag)
    tuner.tune()
    tuner.print_report()

    print("DEEP-LID-TEMPORAL-WALK")
    tuner = DeepLIDTemporalWalk_Tuner(tag, tag)
    tuner.tune()
    tuner.print_report()

    print("DEEP-HUB-TEMPORAL-WALK")
    tuner = DeepHUBTemporalWalk_Tuner(tag, tag)
    tuner.tune()
    tuner.print_report()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: primeval_trw dataset-index")
        sys.exit()
    
    dataset_index = int(sys.argv[1])
    eval_on_dataset(dataset_index)
