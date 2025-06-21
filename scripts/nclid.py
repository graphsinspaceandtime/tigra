"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Script for analyzing NC-LID
"""

import sys

sys.path.append("..")
sys.dont_write_bytecode = True

from data.bank import Databank

databank = Databank()
datasets = databank.get_keys()

from tigra.evaluation.lideval import NCLIDEstimator
from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicGraph

from tigra.evaluation.cumstat import CumulativeCorrelation

from tigra.core.loaders import load_event_sequence

import networkx as nx
from scipy.stats import spearmanr
from numpy import std

from scipy.stats import mannwhitneyu
from statistics import mean, stdev


def count_nc_lid_zero(lid_vals):
    count = 0
    for node in lid_vals:
        if lid_vals[node] == 0:
            count += 1

    return count / len(lid_vals)


def prob_sup(X, Y):
    h = 0
    for x in X:
        for y in Y:
            if x > y:
                h += 1

    total = len(X) * len(Y)
    return h / total


def correlation(dict1, dict2):
    # if dict1.keys() != dict2.keys():
    #     print("WARNING: dictionaries with different keys!!!")

    x, y = [], []

    for n in dict1:
        if n in dict2:
            _x = dict1[n]
            _y = dict2[n]

            x.append(_x)
            y.append(_y)

    return spearmanr(x, y)[0]


def mwu_test_lid_diff(dict_diff, dict2):
    x, y = [], []
    for n in dict_diff:
        diff = dict_diff[n]
        if diff > 0:
            x.append(dict2[n])
        elif diff < 0:
            y.append(dict2[n])

    if len(x) == 0 or len(y) == 0:
        return

    avg_x = round(mean(x), 4)
    avg_y = round(mean(y), 4)

    U, pU = mannwhitneyu(x, y)
    ps1 = round(prob_sup(x, y), 4)
    ps2 = round(prob_sup(y, x), 4)
    # total = len(x) + len(y)
    # x_frac = len(x) / total
    # y_frac = len(y) / total
    acc = "yes" if pU >= 0.05 else "no"

    return (
        str(avg_x)
        + ","
        + str(avg_y)
        + ","
        + str(U)
        + ","
        + str(pU)
        + ","
        + acc
        + ","
        + str(ps1)
        + ","
        + str(ps2)
    )


def mwu_test_lid(lid_dict, metric_dict):
    avg = sum(lid_dict.values()) / len(lid_dict.values())
    x, y = [], []

    for node in lid_dict:
        l = lid_dict[node]
        mval = metric_dict[node]
        if l > avg:
            x.append(mval)
        elif l < avg:
            y.append(mval)

    avg_x = round(mean(x), 4)
    avg_y = round(mean(y), 4)

    U, pU = mannwhitneyu(x, y)
    ps1 = round(prob_sup(x, y), 4)
    ps2 = round(prob_sup(y, x), 4)
    total = len(x) + len(y)
    x_frac = round(len(x) / total, 4)
    y_frac = round(len(y) / total, 4)
    acc = "yes" if pU >= 0.05 else "no"

    return (
        str(avg_x)
        + ","
        + str(avg_y)
        + ","
        + str(x_frac)
        + ","
        + str(y_frac)
        + ","
        + str(U)
        + ","
        + str(pU)
        + ","
        + acc
        + ","
        + str(ps1)
        + ","
        + str(ps2)
    )


def examine_NCLID(dg):
    graphs = dg.get_graph_sequence()

    prev_lid_vals = None

    for i in range(len(graphs)):
        print("\n\nSNAPSHOT", i)
        estimator = NCLIDEstimator(graphs[i].G)
        estimator.estimate_lids()
        lid_vals = estimator.get_lid_values()
        # print(estimator.nc_size_distr_str())
        perfect = count_nc_lid_zero(lid_vals)
        print("perfect = ", perfect)
        if perfect == 1:
            continue

        bc = dict(nx.betweenness_centrality(graphs[i].G))
        cc = dict(nx.closeness_centrality(graphs[i].G))
        ec = dict(nx.eigenvector_centrality(graphs[i].G, max_iter=1000))

        pomg = graphs[i].G
        pomg.remove_edges_from(nx.selfloop_edges(pomg))
        si = dict(nx.core_number(pomg))
        deg = dict(nx.degree(graphs[i].G))

        print("bc = ", correlation(lid_vals, bc))
        print("cc = ", correlation(lid_vals, cc))
        print("ec = ", correlation(lid_vals, ec))
        print("si = ", correlation(lid_vals, si))
        print("dg = ", correlation(lid_vals, deg))

        print("MWU-BC", mwu_test_lid(lid_vals, bc))
        print("MWU-CC", mwu_test_lid(lid_vals, cc))
        print("MWU-EC", mwu_test_lid(lid_vals, ec))
        print("MWU-SI", mwu_test_lid(lid_vals, si))
        print("MWU-DG", mwu_test_lid(lid_vals, deg))

        lid_max = max(lid_vals.values())
        print("\nlid_max = ", lid_max)

        avg = sum(lid_vals.values()) / len(lid_vals.values())
        print("lid_avg = ", avg)

        lid_stdev = std(list(lid_vals.values()))
        print("lid_std = ", lid_stdev)

        print("nc-max, nc-avg = ", estimator.nc_size_stats())

        if prev_lid_vals != None:
            diff_dict = dict()
            diff0, diffP, diffN = 0, 0, 0
            cnt_sur = 0

            for node in lid_vals:
                if node in prev_lid_vals:
                    cnt_sur += 1

                    curr_lid = lid_vals[node]
                    prev_lid = prev_lid_vals[node]

                    diff = curr_lid - prev_lid
                    diff_dict[node] = diff

                    if diff > 0:
                        diffP += 1
                    elif diff < 0:
                        diffN += 1
                    else:
                        diff0 += 1

            print("Dbc = ", correlation(diff_dict, bc))
            print("Dcc = ", correlation(diff_dict, cc))
            print("Dec = ", correlation(diff_dict, ec))
            print("Dsi = ", correlation(diff_dict, si))
            print("Ddg = ", correlation(diff_dict, deg))
            print("diffP = ", diffP / cnt_sur)
            print("diffN = ", diffN / cnt_sur)
            print("diff0 = ", diff0 / cnt_sur)

            print("MWU-BC-D", mwu_test_lid_diff(diff_dict, bc))
            print("MWU-CC-D", mwu_test_lid_diff(diff_dict, cc))
            print("MWU-EC-D", mwu_test_lid_diff(diff_dict, ec))
            print("MWU-SI-D", mwu_test_lid_diff(diff_dict, si))
            print("MWU-DG-D", mwu_test_lid_diff(diff_dict, deg))

        prev_lid_vals = lid_vals


def examine_NCLID_2(dataset, dg):
    sum_lid_vals, num_nodes = 0, 0
    num_zero_lid, max_lid = 0, 0

    c_bc = CumulativeCorrelation()
    c_cc = CumulativeCorrelation()
    c_ec = CumulativeCorrelation()
    c_si = CumulativeCorrelation()
    c_dg = CumulativeCorrelation()

    graphs = dg.get_graph_sequence()
    for i in range(len(graphs)):
        estimator = NCLIDEstimator(graphs[i].G)
        estimator.estimate_lids()
        lid_vals = estimator.get_lid_values()

        sum_lid_vals += sum(lid_vals.values())
        num_zero_lid += count_nc_lid_zero(lid_vals)
        m_lid = max(lid_vals.values())
        if m_lid > max_lid:
            max_lid = m_lid

        num_nodes += len(lid_vals)

        bc = dict(nx.betweenness_centrality(graphs[i].G))
        cc = dict(nx.closeness_centrality(graphs[i].G))
        ec = dict(nx.eigenvector_centrality(graphs[i].G, max_iter=1000))

        pomg = graphs[i].G
        pomg.remove_edges_from(nx.selfloop_edges(pomg))
        si = dict(nx.core_number(pomg))
        deg = dict(nx.degree(graphs[i].G))

        c_bc.add_data(lid_vals, bc)
        c_cc.add_data(lid_vals, cc)
        c_ec.add_data(lid_vals, ec)
        c_si.add_data(lid_vals, si)
        c_dg.add_data(lid_vals, deg)

    avg_lid = sum_lid_vals / num_nodes
    zero_lid_prob = num_zero_lid / num_nodes

    _c_bc = c_bc.get_correlation()
    _c_cc = c_cc.get_correlation()
    _c_ec = c_ec.get_correlation()
    _c_si = c_si.get_correlation()
    _c_dg = c_dg.get_correlation()

    s = (
        dataset
        + ","
        + str(round(avg_lid, 4))
        + ","
        + str(round(max_lid, 4))
        + ","
        + str(round(zero_lid_prob, 6))
        + ","
        + str(round(_c_dg, 4))
        + ","
        + str(round(_c_si, 4))
        + ","
        + str(round(_c_bc, 4))
        + ","
        + str(round(_c_cc, 4))
        + ","
        + str(round(_c_ec, 4))
    )

    print(s)


if __name__ == "__main__":
    print("DATASET,AVG-NC-LID,MAX-NC-LID,ZERO-LID,DEG_C,SI_C,BC_C,CC_C,EC_C")
    for dataset in datasets:
        res = load_event_sequence(dataset)
        if res == None:
            continue

        es, time_res = res
        dg = DynamicGraph()
        dg.init_from_event_sequence(es, time_resolution=time_res, directed=False)
        # dg.info()

        examine_NCLID_2(dataset, dg)
