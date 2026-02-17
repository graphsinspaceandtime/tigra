import sys

sys.path.append("..")
sys.dont_write_bytecode = True

if __name__ == "__main__":
    from data.bank import Databank
    from tigra.core.loaders import load_event_sequence

    databank = Databank()
    es, resol = load_event_sequence("radoslaw-email")
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())

    from tigra.core.dgraph import TemporalAggregateGraph
    tag = TemporalAggregateGraph("radoslaw-email-tag", es)
    print(tag.G)

    print("Nodes")
    print("#nodes", len(tag.G.nodes()))
    for n in tag.G.nodes():
        print(n, tag.G.nodes[n]["timestamp"])

    print("\n\n\n")
    print("Edges")
    print("#edges", len(tag.G.edges()))
    for e in tag.G.edges():
        src, dst = e

        weight = ""
        if "weight" in tag.G.edges[e]:
            weight = tag.G.edges[e]["weight"]

        data = ""
        if "edge_data" in tag.G.edges[e]:
            data = tag.G.edges[e]["edge_data"]

        print(src, dst, len(data))
        #print(src, dst, weight, data)