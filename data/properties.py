import sys
import statistics


sys.path.append("..")

import data.bank as databank
from tigra.core.dgraph import DynamicAggregateGraph, EventSequence


def get_properties(name: databank.Dataset, es: EventSequence):
    # TODO: some might not be directed
    g = DynamicAggregateGraph(name, es, directed=False)
    G = g.get_aggregate_graph().G
    degs = [p[1] for p in G.degree]
    max_deg = max(degs)
    min_deg = min(degs)
    avg_deg = statistics.mean(degs)
    median_deg = statistics.median(degs)
    std_deg = statistics.stdev(degs)
    return (
        G.number_of_nodes(),
        G.number_of_edges(),
        max_deg,
        min_deg,
        avg_deg,
        median_deg,
        std_deg,
    )


def print_properties(key: databank.Dataset):
    es = databank.get_data(key)
    (
        num_of_nodes,
        num_of_edges,
        maximum_degree,
        minimum_degree,
        average_degree,
        median_degree,
        std_degree,
    ) = get_properties(key, es)
    print(
        key,
        num_of_nodes,
        num_of_edges,
        maximum_degree,
        minimum_degree,
        average_degree,
        median_degree,
        std_degree,
    )


if __name__ == "__main__":
    # for key in databank.get_keys():
    #     print_properties(key)
    print_properties("radoslaw-email")
