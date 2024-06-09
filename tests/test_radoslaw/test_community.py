import sys
import os
import time

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.evaluation.lid_eval import LFMnx

from numpy import std
from scipy.stats import skew



def compute_expansion(node, community, graph):
    no_good_links = 0
    no_bad_links = 0
    set_comm = set(community)
    for node in set_comm:
        neigh = graph.neighbors(node)
        for n in neigh:
            if n in set_comm:
                no_good_links += 1
            else:
                no_bad_links += 1
                
    no_good_links /= 2.0
    
    return float(no_bad_links)/float(no_good_links + no_bad_links)


databank = Databank()
ds_name = "radoslaw-email"
es = databank.get_data(ds_name)

dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month", directed=False)
dg.info()

seq = dg.get_graph_sequence()
g = seq[0]

full_local = 0
lfm = LFMnx(g.G, alpha=1.0)
for node in g.G.nodes:
    c1 = lfm.identify_natural_community(node)
    if len(c1) == len(g.G.nodes):
        full_local += 1
    c2 = lfm.identify_natural_community(node, force_stay=False)
    stay = node in c2
    if not stay:
        print("node", node, " left its local c")
    if c1 != c2:
        print("node", node, " has different locals")
        print(len(c1), len(c2), stay)

print("Full local", full_local)

"""
lfm = LFMnx(g.G, alpha=1.0)
for node in g.G.nodes:
    community = lfm.identify_natural_community(node)
    e = compute_expansion(node, community, g.G)
    #print(node, e, len(community), len(g.G.nodes))
    if len(community) > 5 and len(community) < 10:
        print(node)
        print(community)
        print(e)  
        for c in community:
            print(c, " --> ", list(g.G.neighbors(c))) 
        #print(list(g.G.neighbors(node)))
        print()
    
"""

"""
i = 0
nc_sizes = dict()

for g in seq:
    nc_sizes[i] = dict()
    lfm = LFMnx(g.G, alpha=1.0)
    for node in g.G.nodes:
        community = lfm.identify_natural_community(node)
        e = compute_expansion(node, community, g.G)
        nc_size = len(community)
        nc_sizes[i][node] = (nc_size, e)
    i = i + 1
    
    
#print("\n\nNC_SIZES:\n")
#for snap in nc_sizes.keys():
#    print("Snapshot ", snap)
#    print(nc_sizes[snap])



ev_nc_size_file = open(ds_name + "_evolutive_nc_size.csv", "a")

print("\nEVOLUTIVE NC SIZE DIST\n")
ev_nc_size_file.write("Snapshot,MIN_NC,MAX_NC,AVG_NC,SDEV,SKEW\n")
for snap in nc_sizes.keys():
    data = list(nc_sizes[snap].values())
    min_nc = min(data)
    max_nc = max(data)
    avg_nc = (sum(data) / len(data))
    sdev_nc = std(data)
    skew_nc = skew(data)
    ev_nc_size_file.write(str(snap) + "," 
                          + str(min_nc) + "," 
                          + str(max_nc) + "," 
                          + str(avg_nc) + "," 
                          + str(sdev_nc) + "," 
                          + str(skew_nc) + "\n")
print("file ready!")    


print("Natural community size over snapshots\n")

ev_seq_file = open(ds_name + "_evolutive_sequence.csv", "a")

ev_seq_file.write("Node,")
for snap in nc_sizes.keys():
    ev_seq_file.write("Snap_" + str(snap) + ",")
ev_seq_file.write("\n")

for node in g.G.nodes:
    ev_seq_file.write(str(node) + ",")
    for snap in nc_sizes.keys():
        if node in nc_sizes[snap].keys():
            info = nc_sizes[snap][node]
            ns = info[0]
            exp = info[1]
            exp_info = "perfect"
            if exp > 0.5:
                exp_info = "BAD"
            elif exp > 0:
                exp_info = "ok"
            ev_seq_file.write(str(ns) +"[" + exp_info + "]" + ",")
        else:
            ev_seq_file.write(",")
    ev_seq_file.write("\n")
    
print("file ready!")
"""
