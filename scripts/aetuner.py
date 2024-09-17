import sys
import os
import time

from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.hypertuner import AE_Tuner

from tigra.embeddings.utils import n_nodes_to_ae_layers

import numpy as np


def tune_ae(last_graph, dg: DynamicSparseMatrixGraph):

    tuner = AE_Tuner(
        last_graph,
        dg,
        n_units=[[128, 64, 32, 16], [64, 32, 16, 8]],
        look_back=[1, 2, 3, 4, 5],
        beta=np.arange(1, 5, 0.5).tolist(),
        nu1=[1e-6],
        nu2=[1e-6],
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        epochs=10,
    )

    tuner.tune()
    tuner.print_report()


if __name__ == "__main__":
    databank = Databank()
    es = databank.get_data("radoslaw-email")

    dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month")

    tune_ae(dsmg.get_last_matrix_as_graph(), dsmg)
