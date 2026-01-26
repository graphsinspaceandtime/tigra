"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Diffusion prediction evaluation

author: svc@dmi.uns.ac.rs
"""

from abc import ABC, abstractmethod
import random
import numpy as np

from tigra.core.dgraph import DynamicAggregateGraph

from tigra.evaluation.clbase.skl import RF_SKLC, KNN_SKLC, NB_SKLC, SVM_SKLC, LR_SKLC
from tigra.evaluation.clbase.tabpfn import TPFN
from tigra.evaluation.clbase.boost import XGB, CAT, LGBM


class DiffusionProcess(ABC):
    def __init__(self):
        self.initially_active = set()
        self.activations = []

    @abstractmethod
    def simulate_diffusion(self, graph_sequence):
        pass

    def get_activations(self):
        return self.activations

    def get_initially_active(self):
        return self.initially_active


class SI(DiffusionProcess):
    def __init__(self, fraction_activated=0.05, beta=0.02, verbose=False):
        super().__init__()
        self.fraction_activated = fraction_activated
        self.beta = beta
        self.verbose = verbose

    def simulate_diffusion(self, graph_sequence):
        G0 = graph_sequence[0]
        for n in G0.G.nodes():
            if random.random() < self.fraction_activated:
                self.initially_active.add(n)

        if self.verbose:
            print("nodes in G[0]: ", len(G0.G.nodes()))
            print("init active  : ", len(self.initially_active))

        self.active = set(self.initially_active)
        self.activations = []
        all_nodes = set()

        for graph in graph_sequence:
            nodes = graph.G.nodes()
            newly_activated = set()

            for node in nodes:
                all_nodes.add(node)
                if node not in self.active:
                    continue

                for nei in graph.G.neighbors(node):
                    if nei not in self.active and random.random() < self.beta:
                        newly_activated.add(nei)

            for n in newly_activated:
                self.active.add(n)

            self.activations.append(set(self.active))

        if self.verbose:
            for i in range(len(self.activations)):
                print(
                    "Activated at ",
                    i,
                    " -- ",
                    len(self.activations[i]),
                    " -- ",
                    len(graph_sequence[i].G.nodes())
                )
                           
class DiffusionPrediction:
    def __init__(
        self,
        embedding,
        dg,
        diffusion_process: DiffusionProcess,
        negative_sampling=True,
        inbalance_factor=1,
        test_snapshots=1,
        verbose=False,
        seed=0
    ):
        # reproducibility
        random.seed(seed)
        np.random.seed(seed)

        self.embedding = embedding
        self.dg = dg
        self.diffusion_process = diffusion_process
        self.verbose = verbose
        self.negative_sampling = negative_sampling
        self.inbalance_factor = inbalance_factor
        self.test_snapshots = test_snapshots

        if isinstance(dg, DynamicAggregateGraph):
            raise Exception(
                "Diffusion cannot be simulated for DynamicAggregateGraph objects"
            )
        elif len(embedding.embeddings) < len(dg.get_graph_sequence()):
            n = len(dg.get_graph_sequence())
            k = len(embedding.embeddings)
            self.graph_sequence = dg.get_graph_sequence()[n - k : n]
        else:
            self.graph_sequence = dg.get_graph_sequence()

        # simulate diffusion
        self.diffusion_process.simulate_diffusion(self.graph_sequence)
        self.initially_active = self.diffusion_process.get_initially_active()
        self.activations = self.diffusion_process.get_activations()

        if self.verbose:
            print("Activations")
            print(
                "Initial active nodes",
                # sorted(self.initially_active),
                # "#",
                len(self.initially_active),
            )
            for i in range(len(self.activations)):
                print(
                    "Activated at ",
                    i,
                    " -- ",
                    # sorted(self.activations[i]),
                    # "#",
                    len(self.activations[i]),
                )

        #
        # classification models
        #

        self.RF_MODELS = [
            (RF_SKLC(numEstimators=1), "RF-1"),
            (RF_SKLC(numEstimators=2), "RF-2"),
            (RF_SKLC(numEstimators=3), "RF-3"),
            (RF_SKLC(numEstimators=4), "RF-4"),
            (RF_SKLC(numEstimators=5), "RF-5"),
            (RF_SKLC(numEstimators=10), "RF-10"),
            (RF_SKLC(numEstimators=15), "RF-15"),
            (RF_SKLC(numEstimators=20), "RF-20"),
            (RF_SKLC(numEstimators=25), "RF-25"),
            (RF_SKLC(numEstimators=50), "RF-50"),
        ]        

        self.NB_MODELS = [
            (NB_SKLC(), "NB")
        ]

        self.KNN_MODELS = [
            (KNN_SKLC(numNeighbors=1), "KNN-1"),
            (KNN_SKLC(numNeighbors=2), "KNN-2"),
            (KNN_SKLC(numNeighbors=3), "KNN-3"),
            (KNN_SKLC(numNeighbors=4), "KNN-4"),
            (KNN_SKLC(numNeighbors=5), "KNN-5"),
            (KNN_SKLC(numNeighbors=10), "KNN-10"),
            (KNN_SKLC(numNeighbors=15), "KNN-15"),
            (KNN_SKLC(numNeighbors=20), "KNN-20"),
        ]

        self.SVM_MODELS = [
            (SVM_SKLC(kernel="rbf"), "SVM-RBF"),
            (SVM_SKLC(kernel="linear"), "SVM-LIN"),
            (SVM_SKLC(kernel="poly"), "SVM-POLY"),
            (SVM_SKLC(kernel="sigmoid"), "SVM-SIGM")
        ]

        self.LR_MODELS = [
            (LR_SKLC(penalty=None), "LR-NOP"),
            (LR_SKLC(penalty="l1"), "LR-L1"),
            (LR_SKLC(penalty="l2"), "LR-L2"),
            (LR_SKLC(penalty="elasticnet"), "LR-ELNET")
        ]

        self.TABPFN_MODELS = [
            (TPFN(), "TPFN-D")
        ]

        self.XGB_MODELS = [
            (XGB(), "XGB-D")
        ]

        self.CAT_MODELS = [
            (CAT(), "CAT-D")
        ]

        self.LGBM_MODELS = [
            (LGBM(), "LGBM-D")
        ]

        self.ALL_MODELS = self.RF_MODELS + \
            self.NB_MODELS     + \
            self.KNN_MODELS    + \
            self.SVM_MODELS    + \
            self.LR_MODELS     + \
            self.TABPFN_MODELS + \
            self.XGB_MODELS    + \
            self.CAT_MODELS    + \
            self.LGBM_MODELS



    def make_ML_data(self):
        self.training_X, self.training_y = [], []
        self.test_X, self.test_y = [], []

        for i in range(1, len(self.graph_sequence)):
            positive_samples, negative_samples = [], []
            negative_samples_selected, positive_samples_selected = [], []

            prev_activations = self.activations[i - 1]
            curr_activations = self.activations[i]
            new_activations = curr_activations - prev_activations

            prev_emb = self.embedding[i - 1]
            curr_emb = self.embedding[i]

            prev_graph = self.graph_sequence[i - 1].G
            curr_graph = self.graph_sequence[i].G
            for n in prev_graph:
                if n not in curr_graph:
                    continue

                if n in prev_activations:
                    continue

                p_emb, c_emb = prev_emb[n], curr_emb[n]
                x_vector = p_emb + c_emb

                if n in new_activations:
                    positive_samples.append(x_vector)
                else:
                    negative_samples.append(x_vector)

            negative_selection = True
            if self.negative_sampling:
                if (len(positive_samples) < len(negative_samples)):
                    negative_samples_selected = random.sample(
                        negative_samples,
                        int(len(positive_samples) * self.inbalance_factor),
                    )
                else:
                    negative_selection = False
                    positive_samples_selected = random.sample(
                        positive_samples, 
                        int(len(negative_samples) * self.inbalance_factor)
                    )

            if i < len(self.graph_sequence) - self.test_snapshots:
                if self.negative_sampling and negative_selection:
                    self.training_X.extend(negative_samples_selected)
                    self.training_y.extend([0] * len(negative_samples_selected))
                else:
                    self.training_X.extend(negative_samples)
                    self.training_y.extend([0] * len(negative_samples))

                if self.negative_sampling and not negative_selection:
                    self.training_X.extend(positive_samples_selected)
                    self.training_y.extend([1] * len(positive_samples_selected))
                else:
                    self.training_X.extend(positive_samples)
                    self.training_y.extend([1] * len(positive_samples))
            else:
                self.test_X.extend(positive_samples)
                self.test_y.extend([1] * len(positive_samples))
                self.test_X.extend(negative_samples)
                self.test_y.extend([0] * len(negative_samples))

        if self.verbose:
            print("Training -- ", len(self.training_X), len(self.training_y))
            num_positive = len([x for x in self.training_y if x == 1])
            print("Positive training samples", num_positive)
            print("Test     -- ", len(self.test_X), len(self.test_y))
            num_positive = len([x for x in self.test_y if x == 1])
            print("Positive test samples", num_positive)


    def train_and_evaluate_classifiers(self, models):
        best_f1, best_f1_cl, best_f1_prec, best_f1_rec = 0, None, 0, 0

        if self.verbose:
            print("MODEL, ACC, F1, PREC, REC, POS-PREC, POS-REC, NEG-PREC, NEG-REC")

        for model in models:
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

                if self.verbose:
                    print(s)
            except ValueError:
                pass

        return best_f1, best_f1_prec, best_f1_rec, best_f1_cl

    
    def init_evaluation(self):
        self.make_ML_data()

    def evaluate_RF(self):
        return self.train_and_evaluate_classifiers(self.RF_MODELS)
    
    def evaluate_KNN(self):
        return self.train_and_evaluate_classifiers(self.KNN_MODELS)

    def evaluate_NB(self):
        return self.train_and_evaluate_classifiers(self.NB_MODELS)
    
    def evaluate_SVM(self):
        return self.train_and_evaluate_classifiers(self.SVM_MODELS)
    
    def evaluate_LR(self):
        return self.train_and_evaluate_classifiers(self.LR_MODELS)
    
    def evaluate_TABPFN(self):
        return self.train_and_evaluate_classifiers(self.TABPFN_MODELS)
    
    def evaluate_ALL(self):
        return self.train_and_evaluate_classifiers(self.ALL_MODELS)

    def evaluate(self, model_type="ALL"):
        if model_type == "RF":
            return self.evaluate_RF()
        elif model_type == "KNN":
            return self.evaluate_KNN()
        elif model_type == "NB":
            return self.evaluate_NB()
        elif model_type == "SVM":
            return self.evaluate_SVM()
        elif model_type == "LR":
            return self.evaluate_LR()
        elif model_type == "TABPFN":
            return self.evaluate_TABPFN()
        
        return self.evaluate_ALL()
    
    




