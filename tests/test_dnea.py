
import sys
sys.path.append("..")

from data.bank import Databank
from tigra.core.dgraph import DynamicGraph
from tigra.core.dnea import EvolDynamics

DATASET_NAME    = "sx-mathoverflow"
TIME_RESOLUTION = "3-months"

databank = Databank()
es = databank.get_data(DATASET_NAME)

dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution=TIME_RESOLUTION, directed=False)
dg.info()

ed = EvolDynamics(dg)
ed.info()

print("\n--- LOADER WITH POST-PROCESSING")

from tigra.core.loaders import load_event_sequence
es_2, resol = load_event_sequence(DATASET_NAME)

dg = DynamicGraph()
dg.init_from_event_sequence(es_2, time_resolution=resol, directed=False)
dg.info()

ed = EvolDynamics(dg)
ed.info()
