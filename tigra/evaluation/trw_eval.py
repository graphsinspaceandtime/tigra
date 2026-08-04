# applicative evaluation of dynamic graph embedding methods based on temporal random walks

import networkx as nx
import random

from tigra.core.dgraph import TemporalAggregateGraph
from tigra.embeddings.rndw.trndw import DeepLIDTemporalWalk, DeepHubTemporalWalk
from tigra.embeddings.rndw.trndwbase import DeepTemporalWalk

from tigra.evaluation.clbase.skl import RF_SKLC, KNN_SKLC, NB_SKLC, SVM_SKLC, LR_SKLC
from tigra.evaluation.clbase.tabpfn import TPFN
from tigra.evaluation.clbase.boost import XGB, CAT, LGBM

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



# ['radoslaw-email', 'fb-forum', 'fb-messages', 'ia-contacts_hypertext2009', 'ia-enron-employees', 
#  'ia-hospital-ward-proximity-attr', 'ia-primary-school-proximity-attr', 'ia-reality-call', 'CollegeMsg', 'email-Eu-core-temporal', 
#  'sx-askubuntu', 'sx-mathoverflow', 'sx-superuser', 'wiki-talk-temporal', 
#  'synthetic-dba', 'synthetic-dbac', 'synthetic-der', 'synthetic-dpp']

conf_dtw = [10, 10, 25, 25, 50, 25, 200, 10, 10, 10]

# DIM, INVERSE, LOG-SCALING, U

conf_hub = [
    (10, True, False, 0.0)
]

conf_lid = [
    (10, False, False, 0.0)
]

def create_embedding_baseline(tag, dataset_index):
    dtw = DeepTemporalWalk(dim=conf_dtw[dataset_index])
    emb = dtw.construct(tag)[0]
    return emb

def create_embedding_hub(tag, dataset_index):
    c = conf_hub[dataset_index]
    dtw = DeepHubTemporalWalk(dim=c[0], inverse=c[1], log_scaling=c[2], u=c[3])
    emb = dtw.construct(tag)[0]
    return emb

def create_embedding_lid(tag, dataset_index):
    c = conf_hub[dataset_index]
    dtw = DeepLIDTemporalWalk(dim=c[0], inverse=c[1], log_scaling=c[2], u=c[3])
    emb = dtw.construct(tag)[0]
    return emb

class SLPEval:
    def __init__(self, dataset_index, event_sequence, tt_frac=0.8, t_frac=0.8):
        self.dataset_index = dataset_index
        self.event_sequence = event_sequence
        self.tt_frac = tt_frac
        self.t_frac = t_frac
        self.events = self.event_sequence.events
        self.num_events = len(self.events)
        print("SLPEval, #events", len(self.events))

        tt_split_index = int(tt_frac * self.num_events)
        t_split_index = int(t_frac * tt_split_index)
        self.start_events = self.events[:t_split_index]
        self.upd_events = self.events[t_split_index:tt_split_index]
        self.train_events = self.events[:tt_split_index]
        self.test_events = self.events[tt_split_index:]

        print("SLPEval, #start-events", len(self.start_events))
        print("SLPEval, #upd-events", len(self.upd_events))
        print("SLPEval, #train-events", len(self.train_events))
        print("SLPEval, #test-events", len(self.test_events))

        self.upd_links = {}
        for e in self.upd_events:
            src = e.src
            dst = e.dst
            if src > dst:
                src, dst = dst, src

            if src in self.upd_links:
                self.upd_links[src].add(dst)
            else:
                self.upd_links[src] = {dst}

        self.test_links = {}
        for e in self.test_events:
            src = e.src
            dst = e.dst 
            if src > dst:
                src, dst = dst, src

            if src in self.test_links:
                self.test_links[src].add(dst)
            else:
                self.test_links[src] = {dst}

        self.event_sequence.shrink_to(tt_split_index)
        self.TAGUPD = TemporalAggregateGraph("SLPEVAL-TAGUPD", self.event_sequence)

        self.event_sequence.shrink_to(t_split_index)
        self.TAG = TemporalAggregateGraph("SLPEVAL-TAG", self.event_sequence)

        self.init_classification_models()
        self.identify_positive_negative_examples()


    def eval(self):
        print("EVALUATING DEEP-TEMPORAL-WALK")
        self.baseline_emb = create_embedding_baseline(self.TAG, self.dataset_index)
        self.make_ml_data(self.baseline_emb)
        self.train_and_evaluate_classifiers()

        print()
        print("EVALUATING DEEP-HUB-TEMPORAL-WALK")
        self.hub_emb = create_embedding_hub(self.TAG, self.dataset_index)
        self.make_ml_data(self.hub_emb)
        self.train_and_evaluate_classifiers()

        print()
        print("EVALUATING DEEP-LID-TEMPORAL-WALK")
        self.lid_emb = create_embedding_lid(self.TAG, self.dataset_index)
        self.make_ml_data(self.lid_emb)
        self.train_and_evaluate_classifiers()


    def init_classification_models(self):
        self.models = [
            (RF_SKLC(numEstimators=5), "RF-5"),
            (RF_SKLC(numEstimators=10), "RF-10"),
            (RF_SKLC(numEstimators=15), "RF-15"),
            (RF_SKLC(numEstimators=20), "RF-20"),
            (RF_SKLC(numEstimators=25), "RF-25"),
            (RF_SKLC(numEstimators=50), "RF-50"),
            (RF_SKLC(numEstimators=75), "RF-75"),
            (RF_SKLC(numEstimators=100), "RF-100"),
            (NB_SKLC(), "NB"),
            (KNN_SKLC(numNeighbors=1), "KNN-1"),
            (KNN_SKLC(numNeighbors=2), "KNN-2"),
            (KNN_SKLC(numNeighbors=3), "KNN-3"),
            (KNN_SKLC(numNeighbors=4), "KNN-4"),
            (KNN_SKLC(numNeighbors=5), "KNN-5"),
            (KNN_SKLC(numNeighbors=10), "KNN-10"),
            (KNN_SKLC(numNeighbors=15), "KNN-15"),
            (KNN_SKLC(numNeighbors=20), "KNN-20"),
            (KNN_SKLC(numNeighbors=25), "KNN-25"),
            (SVM_SKLC(kernel="rbf"), "SVM-RBF"),
            (SVM_SKLC(kernel="linear"), "SVM-LIN"),
            (SVM_SKLC(kernel="poly"), "SVM-POLY"),
            (SVM_SKLC(kernel="sigmoid"), "SVM-SIGM"),
            (LR_SKLC(penalty=None), "LR-NOP"),
            (LR_SKLC(penalty="l1"), "LR-L1"),
            (LR_SKLC(penalty="l2"), "LR-L2"),
            (LR_SKLC(penalty="elasticnet"), "LR-ELNET"),
            (TPFN(), "TPFN"),
            (XGB(), "XGB"),
            (CAT(), "CAT"),
            (LGBM(), "LGBM")
        ]


    def identify_positive_negative_examples(self):
        g = self.TAG.G
        self.positive_train_pairs, self.negative_train_pairs = [], []
        
        for u, v in nx.non_edges(g):
            if u > v:
                u, v = v, u

            if u in self.upd_links and v in self.upd_links[u]:
                self.positive_train_pairs.append((u, v))
            else:
                self.negative_train_pairs.append((u, v))

        print("SLPEval, #positive-train-pairs", len(self.positive_train_pairs))
        print("SLPEval, #negative-train-pairs", len(self.negative_train_pairs))

        g_upd = self.TAGUPD.G

        self.positive_test_pairs, self.negative_test_pairs = [], []
        for u, v in nx.non_edges(g):
            if u > v:
                u, v = v, u
        
            if g_upd.has_edge(u, v):
                continue

            if u in self.test_links and v in self.test_links[u]:
                self.positive_test_pairs.append((u, v))
            else:
                self.negative_test_pairs.append((u, v))

        print("SLPEval, #positive-test-pairs", len(self.positive_test_pairs))
        print("SLPEval, #negative-test-pairs", len(self.negative_test_pairs))

        print("... negative-sampling")
        self.negative_train_pairs = random.sample(self.negative_train_pairs, k=len(self.positive_train_pairs))
        self.negative_test_pairs = random.sample(self.negative_test_pairs, k=len(self.positive_test_pairs))
        print("SLPEval, #negative-train-pairs", len(self.negative_train_pairs))
        print("SLPEval, #negative-test-pairs", len(self.negative_test_pairs))


    def make_ml_data(self, embedding):
        self.training_X, self.training_y = [], []
        self.test_X, self.test_y = [], []

        for tp in self.positive_train_pairs:
            src, dst = tp[0], tp[1]
            emb1 = embedding[src]
            emb2 = embedding[dst]
            e = emb1 + emb2
            self.training_X.append(e)
            self.training_y.append(1)

        for tp in self.negative_train_pairs:
            src, dst = tp[0], tp[1]
            emb1 = embedding[src]
            emb2 = embedding[dst]
            e = emb1 + emb2
            self.training_X.append(e)
            self.training_y.append(0)

        for tp in self.positive_test_pairs:
            src, dst = tp[0], tp[1]
            emb1 = embedding[src]
            emb2 = embedding[dst]
            e = emb1 + emb2
            self.test_X.append(e)
            self.test_y.append(1)

        for tp in self.negative_test_pairs:
            src, dst = tp[0], tp[1]
            emb1 = embedding[src]
            emb2 = embedding[dst]
            e = emb1 + emb2
            self.test_X.append(e)
            self.test_y.append(0)


    def train_and_evaluate_classifiers(self):
        best_f1, best_f1_cl, best_f1_prec, best_f1_rec = 0, None, 0, 0
    
        print("MODEL, ACC, F1, PREC, REC, POS-PREC, POS-REC, NEG-PREC, NEG-REC")
    
        for model in self.models:
            try:
                sk_model = model[0]
                model_name = model[1]
        
                f1, acc, prec, rec, pos_prec, pos_rec, neg_prec, neg_rec = sk_model.evaluate_binarycl(
                    self.training_X, self.training_y, self.test_X, self.test_y
                )
    
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_cl = model_name
                    best_f1_prec = prec
                    best_f1_rec = rec
    
                s = model_name + ","
                s += str(round(acc, 4)) + ","
                s += str(round(f1, 4)) + ","
                s += str(round(prec, 4)) + ","
                s += str(round(rec, 4)) + ","
                s += str(round(pos_prec, 4)) + ","
                s += str(round(pos_rec, 4)) + ","
                s += str(round(neg_prec, 4)) + ","
                s += str(round(neg_rec, 4))
                print(s)
            except ValueError:
                pass

        print("BEST")
        print(best_f1, best_f1_cl)
        return best_f1, best_f1_prec, best_f1_rec, best_f1_cl

        
            




        