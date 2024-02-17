import sys

sys.path.append("../")
sys.dont_write_bytecode = True

from tigra.core.loaders import load_netrep_dataset
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.sota.stwalk import STWalk

es = load_netrep_dataset("../data/netrep/ia-radoslaw-email.edges")
print(es)
dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False)
dg.info()
ts = len(dg.get_nx_graph_sequence())
st = STWalk(number_restart=40, walk_length=10, dimension=5, time_step=ts,time_window_size=1,workers=4,vocab_window_size=5)
st.make_embedding(dg)
for i in range(0, len(st.embeddings)):
    print(st.embeddings[i])
    
print(len(st.embeddings))