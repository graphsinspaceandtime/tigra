"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for computing dynamic network statistics
"""

import sys
import os

sys.path.append("..")

from tigra.core.dgraph import DynamicGraph
from tigra.core.dnea import EvolDynamics
from tigra.core.loaders import load_event_sequence, datasets

for dataset in datasets:
    print(dataset)
    es, resol = load_event_sequence(dataset)
    print("#events", es.num_events())

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()

    ed = EvolDynamics(dg)
    ed.info()

    print()