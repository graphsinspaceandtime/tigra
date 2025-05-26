import sys

sys.path.append("..")

from tigra.embeddings.sota.factory import get_deep_hub_factory

from tigra.core.greval import GraphReconstructionEval

deep_hub_fact = get_deep_hub_factory("../deep_hub.conf")

from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

for dataset in datasets:
    print("DATASET", dataset)
    dn2v, dg = deep_hub_fact.get_embedding(dataset)

    org_graphs = dg.get_graph_sequence()
    num_graphs = len(org_graphs)

    for i in range(num_graphs):
        og = org_graphs[i]
        emb = dn2v.embeddings[i]

        print("#nodes", og.num_nodes(), "#links", og.num_edges())
        num_links = og.num_edges()

        rg = emb.reconstruct(num_links, "REC")
        print("#nodes", rg.num_nodes(), "#links", rg.num_edges())

        gre = GraphReconstructionEval(og, emb, report_warnings=False)
        print("Snapshot", i)
        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())
        print()

    print("----------------------------------------------------------")
