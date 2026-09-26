# applicative evaluation of dynamic graph embedding methods based on temporal random walks

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import networkx as nx
import numpy as np
import random

from tigra.core.dgraph import TemporalAggregateGraph
from tigra.core.dgraph import DynamicAggregateGraph
from tigra.embeddings.rndw.trndw import DeepLIDTemporalWalk, DeepHubTemporalWalk
from tigra.embeddings.rndw.trndwbase import DeepTemporalWalk
from tigra.embeddings.sota.ctdne_wrapper import Ctdne

from tigra.evaluation.clbase.skl import RF_SKLC, KNN_SKLC, NB_SKLC, SVM_SKLC, LR_SKLC
from tigra.evaluation.clbase.tabpfn import TPFN
from tigra.evaluation.clbase.boost import XGB, CAT, LGBM

from sklearn.metrics import roc_auc_score, average_precision_score

from networkx.algorithms.link_prediction import jaccard_coefficient, adamic_adar_index


from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import louvain_communities

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


# datasets
# ['radoslaw-email', 'fb-forum', 'fb-messages', 'ia-contacts_hypertext2009', 'ia-enron-employees', 
#  'ia-hospital-ward-proximity-attr', 'ia-primary-school-proximity-attr', 'ia-reality-call', 'CollegeMsg', 'email-Eu-core-temporal', 
#  'sx-askubuntu', 'sx-mathoverflow', 'sx-superuser', 'wiki-talk-temporal', 
#  'synthetic-dba', 'synthetic-dbac', 'synthetic-der', 'synthetic-dpp']


# configurations

conf_dtw = [10, 10, 25, 25, 50, 25, 200, 10, 10, 10]

conf_hub = [
    (10, True, False, 0.0),   # radoslaw
    (25, True, False, 0.0),   # fb-forum
    (25, True, False, 0.15),  # fb-messages
    (50, True, False, 0.0),   # ia-contacts
    (25, True, False, 0.0),   # ia-enron
    (25, True, False, 0.0),   # ia-hospital
    (50, True, False, 0.15),  # ia-primary
    (-1, -1, -1, -1),         # ia-reality
    (10, True, False, 0.0),   # college-msg
    (25, True, False, 0.0)    # email-eu
]

conf_lid = [
    (10, False, False, 0.0),  # radoslaw
    (10, False, True, 0.5),   # fb-forum
    (25, True, True, 0.75),   # fb-messages
    (25, False, False, 0.25), # ia-contacts
    (50, True, False, 0.15),  # ia-enron
    (25, False, False, 0.15), # ia-hospital
    (100, False, False, 0.75),# ia-primary
    (-1, -1, -1, -1),         # ia-reality
    (10, True, False, 0.5),   # college-msg
    (10, False, False, 0,0)   # email-eu
]

conf_ctdne = [
    (200, 4.0, 0.5),
    (10, 4.0, 2.0),
    (25, 0.5, 4.0),
    (10, 4.0, 1.0),
    (10, 1.0, 4.0),
    (50, 2.0, 0.5),
    (25, 2.0, 4.0),
    (-1, -1, -1),
    (50, 0.5, 2.0),
    (200, 2.0, 0.5)
]

# embedding factory functions
def create_embedding_ctdne(tag, dataset_index):
    c = conf_ctdne[dataset_index]
    ctdne = Ctdne(dim=c[0], p=c[1], q=c[2], walk_length=32, num_walks=10, workers=1)
    emb = ctdne.construct(tag)[0]
    return emb

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

def get_classification_models():
    return [
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
        #(TPFN(), "TPFN"),
        #(XGB(), "XGB"),
        #(CAT(), "CAT"),
        #(LGBM(), "LGBM")
    ]

# Base class for link prediction
class LPBase:
    def __init__(self, dataset_index, event_sequence, tt_frac=0.8):
        self.dataset_index = dataset_index
        self.event_sequence = event_sequence
        self.events = self.event_sequence.events
        self.num_events = len(self.events)
        print("LP-Eval, #events", len(self.events))

        self.tt_frac = tt_frac
        self.tt_split_index = int(tt_frac * self.num_events)

        self.train_events = self.events[:self.tt_split_index]
        self.test_events = self.events[self.tt_split_index:]

        print("LP-Eval, #train-events", len(self.train_events))
        print("LP-Eval, #test-events", len(self.test_events))

        self.init_test_links()


    def init_test_links(self):
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



# supervised link prediction
class SLPEval(LPBase):
    def __init__(self, dataset_index, event_sequence, tt_frac=0.8, t_frac=0.8):
        super().__init__(dataset_index, event_sequence, tt_frac=tt_frac)
        self.t_frac = t_frac        
        t_split_index = int(t_frac * self.tt_split_index)
        self.start_events = self.events[:t_split_index]
        self.upd_events = self.events[t_split_index:self.tt_split_index]
        
        print("SLPEval, #start-events", len(self.start_events))
        print("SLPEval, #upd-events", len(self.upd_events))
        
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

        self.event_sequence.shrink_to(self.tt_split_index)
        self.TAGUPD = TemporalAggregateGraph("SLPEVAL-TAGUPD", self.event_sequence)

        self.event_sequence.shrink_to(t_split_index)
        self.TAG = TemporalAggregateGraph("SLPEVAL-TAG", self.event_sequence)
        self.DAG = DynamicAggregateGraph("DAG", self.event_sequence, directed=False)

        self.init_classification_models()
        self.identify_positive_negative_examples()


    def eval(self):
        #print("EVALUATING CTDNE")
        #self.ctdne_emb = create_embedding_ctdne(self.DAG, self.dataset_index)
        #self.make_ml_data(self.ctdne_emb)
        #self.train_and_evaluate_classifiers()

        print()
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
        self.models = get_classification_models()


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


# unsupervised link prediction        
class ULPEval(LPBase):
    def __init__(self, dataset_index, event_sequence, tt_frac=0.8):
        super().__init__(dataset_index, event_sequence, tt_frac=tt_frac)
        self.event_sequence.shrink_to(self.tt_split_index)
        self.TAG = TemporalAggregateGraph("TAG", self.event_sequence)
        self.DAG = DynamicAggregateGraph("DAG", self.event_sequence, directed=False) 

        g = self.TAG.G
        links_created = 0
        for e in self.test_events:
            src, dst = e.src, e.dst
            if not g.has_edge(src, dst):
                links_created += 1

        self.K = links_created

        print("ULPEVAL, #links-created", links_created)    

    def compute_scores(self, similarity, labels):
        ss, sl = map(list, zip(*sorted(zip(similarity, labels), reverse=True)))
        ss = ss[:self.K]
        sl = sl[:self.K]

        correct = sl.count(1)
        acc = correct / self.K

        ra = roc_auc_score(sl, ss)
        ap = average_precision_score(sl, ss)
        return acc, ra, ap

    def compute_sim_scores(self, embedding):
        g = self.TAG.G

        cos_sim, euc_sim, l1_sim, labels = [], [], [], []

        for u, v in nx.non_edges(g):
            if u > v:
                u, v = v, u

            emb1 = embedding[u]
            emb2 = embedding[v]

            cs = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            cos_sim.append(cs)
            
            es = float(-np.linalg.norm(np.array(emb1) - np.array(emb2)))
            euc_sim.append(es)

            l1 = float(-np.sum(np.abs(np.array(emb1) - np.array(emb2))))
            l1_sim.append(l1)

            link_created = u in self.test_links and v in self.test_links[u]
            labels.append(1 if link_created else 0)

        acc, ra, ap = self.compute_scores(cos_sim, labels)    
        print("COS-SIM," + str(acc) + "," + str(ra) + "," + str(ap))

        acc, ra, ap = self.compute_scores(euc_sim, labels)
        print("EUC-SIM," + str(acc) + "," + str(ra) + "," + str(ap))

        acc, ra, ap = self.compute_scores(l1_sim, labels)
        print("L1--SIM," + str(acc) + "," + str(ra) + "," + str(ap))
        
    def compute_structural_metrics(self):
        g = self.TAG.G

        labels = []
        for u, v in nx.non_edges(g):
            if u > v:
                u, v = v, u

            link_created = u in self.test_links and v in self.test_links[u]
            labels.append(1 if link_created else 0)

        aa = nx.adamic_adar_index(g, nx.non_edges(g))
        aa = [item[2] for item in aa]
        acc, ra, ap = self.compute_scores(aa, labels)    
        print("AA--SIM," + str(acc) + "," + str(ra) + "," + str(ap))

        js = nx.jaccard_coefficient(g, nx.non_edges(g))
        js = [item[2] for item in js]
        acc, ra, ap = self.compute_scores(js, labels)    
        print("JS--SIM," + str(acc) + "," + str(ra) + "," + str(ap))    
    
    def eval(self):
        print("STRUCTURAL METRICS")
        self.compute_structural_metrics()

        print()
        print("EVALUATING CTDNE")
        self.ctdne_emb = create_embedding_ctdne(self.DAG, self.dataset_index)
        self.compute_sim_scores(self.ctdne_emb)

        print()
        print("EVALUATING DEEP-TEMPORAL-WALK")
        self.baseline_emb = create_embedding_baseline(self.TAG, self.dataset_index)
        self.compute_sim_scores(self.baseline_emb)
        
        print()
        print("EVALUATING DEEP-HUB-TEMPORAL-WALK")
        self.hub_emb = create_embedding_hub(self.TAG, self.dataset_index)
        self.compute_sim_scores(self.hub_emb)
        
        print()
        print("EVALUATING DEEP-LID-TEMPORAL-WALK")
        self.lid_emb = create_embedding_lid(self.TAG, self.dataset_index)
        self.compute_sim_scores(self.lid_emb)


# classification evaluation
# ia-hospital-ward-proximity-attr    --> 5
# ia-primary-school-proximity-attr   --> 6
class CLEval:
    def __init__(self, dataset_index, event_sequence):
        self.dataset_index = dataset_index
        self.models = get_classification_models()
        self.TAG = TemporalAggregateGraph("TAG", event_sequence)
        self.DAG = DynamicAggregateGraph("DAG", event_sequence, directed=False)

        self.label_dict = {}
        num_label_warns = 0
        classes = set()

        for e in event_sequence:
            src, dst = e.src, e.dst
            if (
                e.event_data != None
                and e.event_data["src_class"] != None
                and e.event_data["dst_class"] != None
            ):
                src_label = e.event_data["src_class"]
                dst_label = e.event_data["dst_class"]

                classes.add(src_label)
                classes.add(dst_label)
                
                if src in self.label_dict:
                    old_src_label = self.label_dict[src]
                    if src_label != old_src_label:
                        print("[Warning, label]", src, old_src_label, src_label)
                        num_label_warns += 1

                if dst in self.label_dict:
                    old_dst_label = self.label_dict[dst]
                    if dst_label != old_dst_label:
                        print("[Warning, label]", dst, old_dst_label, dst_label)
                        num_label_warns += 1
                    
                self.label_dict[src] = e.event_data["src_class"]
                self.label_dict[dst] = e.event_data["dst_class"]

        print("# labeled nodes", len(self.label_dict))
        print("# events", event_sequence.num_events())
        print("# TAG nodes", self.TAG.G.number_of_nodes())
        print("# DAG nodes", self.DAG.G.number_of_nodes())
        print("# label warnings", num_label_warns)
        print("# classes", classes)

    def make_ml_data(self, embedding):
        node_ids = embedding.get_node_ids()
        self.node_vectors = [embedding[node_id] for node_id in node_ids]
        self.labels = [self.label_dict[node_id] for node_id in node_ids]

    def train_and_evaluate_classifiers(self):
        best_f1, best_f1_cl, best_f1_prec, best_f1_rec = 0, None, 0, 0
        
        print("MODEL, ACC, F1, PREC, REC")
        print(len(self.models))
        
        for model in self.models:
            try:
                sk_model = model[0]
                model_name = model[1]
            
                #f1, acc, prec, rec, pos_prec, pos_rec, neg_prec, neg_rec = sk_model.evaluate_binarycl(
                #        self.training_X, self.training_y, self.test_X, self.test_y
                #    )

                acc, prec, rec, f1 = sk_model.evaluate_kfold_cv(self.node_vectors, self.labels, n_splits=5)
        
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_cl = model_name
                    best_f1_prec = prec
                    best_f1_rec = rec
        
                s = model_name + ","
                s += str(round(acc, 4)) + ","
                s += str(round(f1, 4)) + ","
                s += str(round(prec, 4)) + ","
                s += str(round(rec, 4))
                
                print(s)
            except Exception:
                pass
    
        print("BEST")
        print(best_f1, best_f1_cl)
        return best_f1, best_f1_prec, best_f1_rec, best_f1_cl

    def eval(self):
        #print("EVALUATING CTDNE")
        #self.ctdne_emb = create_embedding_ctdne(self.DAG, self.dataset_index)
        #self.make_ml_data(self.ctdne_emb)
        #self.train_and_evaluate_classifiers()
    
        print()
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


# clustering evaluation
class ClusteringEval:
    def __init__(self, dataset_index, event_sequence):
        self.dataset_index = dataset_index
        self.TAG = TemporalAggregateGraph("TAG", event_sequence)

        #lc_partition = louvain_communities(self.TAG.G)
        #print("#communities", len(lc_partition))
        #q = modularity(self.TAG.G, lc_partition)
        #print("LOUV, Q", q)

    def eval_kmeans(self, emb_base, emb_hub, emb_lid):
        i_base, i_hub, i_lid = [], [], []

        print("Silhouette score")
        print("NUM-CLUSTERS,DEEP-TW,DEEP-HUB-TW,DEEP-LID-TW")
        for num_clusters in range(2, 21):
            node_ids = emb_base.get_node_ids()
            node_vectors = np.array([emb_base[node] for node in node_ids])
            km = KMeans(random_state=42, n_init="auto", n_clusters=num_clusters).fit(node_vectors)
            labels = km.labels_
            ss_base = silhouette_score(node_vectors, labels)
            i_base.append(km.inertia_)

            node_ids = emb_hub.get_node_ids()
            node_vectors = np.array([emb_hub[node] for node in node_ids])
            km = KMeans(random_state=42, n_init="auto", n_clusters=num_clusters).fit(node_vectors)
            labels = km.labels_
            ss_hub = silhouette_score(node_vectors, labels)
            i_hub.append(km.inertia_)

            node_ids = emb_lid.get_node_ids()
            node_vectors = np.array([emb_lid[node] for node in node_ids])
            km = KMeans(random_state=42, n_init="auto", n_clusters=num_clusters).fit(node_vectors)
            labels = km.labels_
            ss_lid = silhouette_score(node_vectors, labels)
            i_lid.append(km.inertia_)

            s = str(num_clusters) + "," + str(round(ss_base, 4)) + "," + str(round(ss_hub, 4)) + "," + str(round(ss_lid, 4))
            print(s)

        kl = KneeLocator(list(range(2, 21)), i_base, curve="convex", direction="decreasing")
        print("BASE K", kl.elbow)

        kl = KneeLocator(list(range(2, 21)), i_hub, curve="convex", direction="decreasing")
        print("HUB K", kl.elbow)

        kl = KneeLocator(list(range(2, 21)), i_lid, curve="convex", direction="decreasing")
        print("LID K", kl.elbow)

        print("Inertia")
        print("NUM-CLUSTERS,DEEP-TW,DEEP-HUB-TW,DEEP-LID-TW")
        for i in range(2, 21):
            s = str(i) + "," + str(round(i_base[i - 2], 4)) + "," + str(round(i_hub[i - 2], 4)) + "," + str(round(i_lid[i - 2], 4))
            print(s)


    def eval(self):
        baseline_emb = create_embedding_baseline(self.TAG, self.dataset_index)
        hub_emb = create_embedding_hub(self.TAG, self.dataset_index)
        lid_emb = create_embedding_lid(self.TAG, self.dataset_index)
        self.eval_kmeans(baseline_emb, hub_emb, lid_emb)

        
        """
        node_ids = self.baseline_emb.get_node_ids()
        node_vectors = np.array([self.baseline_emb[node] for node in node_ids])

        num_clusters = 2
        for num_clusters in range(2, 15):
            print(num_clusters)
            km = KMeans(random_state=42, n_init="auto", n_clusters=num_clusters).fit(node_vectors)
            labels = km.labels_
            print(labels)
            ss = silhouette_score(node_vectors, labels)
            print("SS", ss)

            partition = list()
            for i in range(num_clusters):
                partition.append(set())

            for i in range(len(node_ids)):
                nid = node_ids[i]
                lab = labels[i]
                partition[lab].add(nid)

            q = modularity(self.TAG.G, partition)
            print("Q", q)
            print()
        """

def js(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    
    # Handle the empty set edge case to prevent ZeroDivisionError
    if not s1 and not s2:
        return 1.0
        
    actual_intersection = len(s1.intersection(s2))
    actual_union = len(s1.union(s2))
    
    return actual_intersection / actual_union

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

class AnomEval:
    def __init__(self, dataset_index, event_sequence):
        self.dataset_index = dataset_index
        self.TAG = TemporalAggregateGraph("TAG", event_sequence)

    def detect_anomalies(self, emb):
        node_ids = emb.get_node_ids()
        node_vectors = np.array([emb[node] for node in node_ids])
        clf = OneClassSVM(gamma='auto').fit(node_vectors)
        oc_outliers = clf.predict(node_vectors)
        #print(oc_outliers)

        clf = LocalOutlierFactor(contamination=0.05)
        lof_outliers = clf.fit_predict(node_vectors)
        #print(lof_outliers)

        clf = IsolationForest(random_state=0, contamination=0.05).fit(node_vectors)
        if_outliers = clf.predict(node_vectors)
        #print(if_outliers)

        cov = EllipticEnvelope(random_state=0, contamination=0.05).fit(node_vectors)
        ee_outliers = cov.predict(node_vectors)

        strong_evidence, medium_evidence = [], []

        print(len(node_ids))
        for i in range(len(node_ids)):
            node_id = node_ids[i]
            a1 = 1 if oc_outliers[i] == -1 else 0
            a2 = 1 if lof_outliers[i] == -1 else 0
            a3 = 1 if if_outliers[i] == -1 else 0
            a4 = 1 if ee_outliers[i] == -1 else 0

            score = a1 + a2 + a3 + a4

            if score == 4:
                strong_evidence.append(node_id)
            elif score == 3:
                medium_evidence.append(node_id)

        return strong_evidence


    def eval(self):
        baseline_emb = create_embedding_baseline(self.TAG, self.dataset_index)
        hub_emb = create_embedding_hub(self.TAG, self.dataset_index)
        lid_emb = create_embedding_lid(self.TAG, self.dataset_index)

        nodes = self.TAG.G.number_of_nodes()

        #print("DEEP-TW")
        s1 = self.detect_anomalies(baseline_emb)
        #print(len(s1))
        #print(s1)

        #print("DEEP-HUB-TW")
        s2 = self.detect_anomalies(hub_emb)
        #print(len(s2))
        #print(s2)

        #print("DEEP-LID-TW")
        s3 = self.detect_anomalies(lid_emb)
        #print(len(s3))
        #print(s3)

        j12, j13, j23 = js(s1, s2), js(s1, s3), js(s2, s3)

        #print(len(s1), len(s2), len(s3), j12, j13, j23)
        
        s = str(nodes) + "," + str(len(s1)) + "," + str(len(s2)) + "," + str(len(s3)) + ","
        s += str(round(j12, 4)) + "," + str(round(j13, 4)) + "," + str(round(j23, 4))
        print(s)
