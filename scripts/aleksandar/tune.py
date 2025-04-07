import sys
import os

sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..", ".."))

from rwtuner import tune_hub_changed_prop, tune_hub_changed_reverse_prop
from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.loaders import load_event_sequence


if __name__ == "__main__":

    """
    radoslaw-email,
    fb-forum,
    fb-messages,
    ia-contacts_hypertext2009,
    ia-enron-employees,
    ia-hospital-ward-proximity-attr,
    ia-primary-school-proximity-attr,
    ia-reality-call,
    CollegeMsg,
    email-Eu-core-temporal
    """
    dataset_name = sys.argv[1].strip()

    es, time_resolution = load_event_sequence(dataset_name)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=time_resolution, directed=False)
    dg.info()

    org_graphs = dg.get_graph_sequence()
    last_graph = org_graphs[-1]

    print("\nTUNNING", dataset_name)
    tune_hub_changed_prop(last_graph, dg)
    print("\##########")
    tune_hub_changed_reverse_prop(last_graph, dg)

