
import sys
sys.path.append("..")

from data.bank import Databank
from tigra.core.dgraph import DynamicGraph
from tigra.core.dnea import EvolDynamics

databank = Databank()
es = databank.get_data("radoslaw-email")

dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False)
dg.info()

ed = EvolDynamics(dg)
ed.info()

