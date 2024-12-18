import sys

sys.path.append("../../..")

from data.bank import Databank
from tigra.core.dgraph import DynamicGraph
import numpy as np
import matplotlib.pyplot as plt

db = Databank()


def generate_image(dataset):
    es = db.get_data(dataset)
    meta = db.get_metadata(dataset)

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        es,
        time_resolution=meta["resolution"],
        directed=False,
        cumulative=meta["cumulative"],
    )

    print("Generating " + dataset)

    values = [g._degree_stats() for g in dg.get_graph_sequence()]

    values1 = [value[0] for value in values]
    values2 = [value[1] for value in values]
    values3 = [value[2] for value in values]

    snaps = np.arange(1, len(values) + 1)
    x_positions = np.arange(1, len(values) + 1) * 1.25
    str_snaps = [str(snap) for snap in snaps]

    width = 0.35

    plt.figure(figsize=(426 / 72, 213 / 72))

    plt.bar(x_positions - width, values1, width, label="Mean")
    plt.bar(x_positions, values2, width, label="Median")
    plt.bar(x_positions + width, values3, width, label="Std. dev.")

    plt.xlabel("Time window")
    plt.ylabel("Value")

    plt.title(dataset)
    plt.xticks(x_positions, str_snaps)
    plt.legend()

    plt.savefig("imgs/" + dataset + ".png", dpi=300, bbox_inches="tight")


for dataset in db.get_keys():
    generate_image(dataset)
