"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Cumulative statistics and statistical tests

author: svc@dmi.uns.ac.rs, lucy@dmi.uns.ac.rs
"""

from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
from statistics import mean

def correlation(dict1, dict2):
    x, y = [], []

    for n in dict1:
        if n in dict2:
            _x = dict1[n]
            _y = dict2[n]

            x.append(_x)
            y.append(_y)

    return spearmanr(x, y)[0]


class CumulativeStatistics:
    def __init__(self):
        self.x = []
        self.y = []


    def add_data(self, dict1, dict2):
        for n in dict1:
            if n in dict2:
                _x = dict1[n]
                _y = dict2[n]

                self.x.append(_x)
                self.y.append(_y)


    def add_high_low_groups(self, xdict, ydict):
        avg = sum(xdict.values()) / len(xdict.values())
        
        for node in xdict:
            l = xdict[node]
            if node in ydict:
                mval = ydict[node]
                if l > avg:
                    self.x.append(mval)
                elif l < avg:
                    self.y.append(mval)



class CumulativeCorrelation(CumulativeStatistics):
    def __init__(self):
        super().__init__()

    def get_correlation(self):
        return spearmanr(self.x, self.y)[0]


def prob_sup(X, Y):
    h = 0
    for x in X:
        for y in Y:
            if x > y:
                h += 1

    total = len(X) * len(Y)
    return h / total


class CumulativeMWU(CumulativeStatistics):
    def __init__(self):
        super().__init__()

    def mwu(self):
        avg_x = round(mean(self.x), 4)
        avg_y = round(mean(self.y), 4)

        U, pU = mannwhitneyu(self.x, self.y)
        ps1 = round(prob_sup(self.x, self.y), 4)
        ps2 = round(prob_sup(self.y, self.x), 4)
        total = len(self.x) + len(self.y)
        x_frac = round(len(self.x) / total, 4)
        y_frac = round(len(self.y) / total, 4)
        acc = "yes" if pU >= 0.05 else "no"

        return str(avg_x) + "," + str(avg_y) + "," + str(x_frac) + "," + str(y_frac) + "," +\
            str(U) + "," + str(pU) + "," + acc + "," + str(ps1) + "," + str(ps2)