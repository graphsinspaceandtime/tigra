import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.loaders import load_event_sequence


from tigra.evaluation.linkpred import (
    EuclideanDistance,
    UnsupervisedLinkPrediction,
    Predicter,
)

from tigra.evaluation.lideval import NCLIDEstimator
from tigra.embeddings.ae.autoencoders import DistanceDynAE


class CappedDistanceDynAE(DistanceDynAE):
    def __init__(
        self, dim=32, hidden_dim=64, epochs=300, lr=0.01, device=None, pos_cap=None
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device)
        self.pos_cap = pos_cap

    def compute_loss(self, recon_logits, target_adj, G, nodelist):
        num_edges = target_adj.sum().item()
        num_nodes = target_adj.shape[0]
        raw_pos_weight = (num_nodes**2 - num_edges) / (num_edges + 1e-9)
        pos_weight_val = (
            min(raw_pos_weight, self.pos_cap) if self.pos_cap else raw_pos_weight
        )

        base_weights = torch.ones_like(target_adj)
        base_weights[target_adj == 1] = pos_weight_val
        return (
            F.binary_cross_entropy_with_logits(
                recon_logits, target_adj, reduction="none"
            )
            * base_weights
        ).mean()


class HubDistanceDynAE(CappedDistanceDynAE):
    def __init__(
        self,
        dim=32,
        hidden_dim=64,
        epochs=300,
        lr=0.01,
        device=None,
        pos_cap=None,
        hub_alpha=1.0,
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device, pos_cap)
        self.hub_alpha = hub_alpha

    def compute_loss(self, recon_logits, target_adj, G, nodelist):
        base_loss = super().compute_loss(recon_logits, target_adj, G, nodelist)
        degrees = torch.tensor(
            [G.degree(n) for n in nodelist], device=self.device, dtype=torch.float32
        )
        deg_outer = torch.outer(degrees, degrees)
        normalized_hub = deg_outer / (deg_outer.max() + 1e-9)
        hub_multiplier = 1.0 + (self.hub_alpha * normalized_hub)
        return (base_loss * hub_multiplier).mean()


class LIDDistanceDynAE(CappedDistanceDynAE):
    def __init__(
        self,
        dim=32,
        hidden_dim=64,
        epochs=300,
        lr=0.01,
        device=None,
        pos_cap=None,
        lid_alpha=1.0,
        precomputed_lids=None,
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device, pos_cap)
        self.lid_alpha = lid_alpha
        self.precomputed_lids = precomputed_lids
        self.current_snap_idx = 0

    def train_snapshot(self, G):
        self.current_lid_tensor = self.precomputed_lids[self.current_snap_idx].to(
            self.device
        )
        self.current_snap_idx += 1
        return super().train_snapshot(G)

    def compute_loss(self, recon_logits, target_adj, G, nodelist):
        base_loss = super().compute_loss(recon_logits, target_adj, G, nodelist)
        lid_outer = torch.outer(self.current_lid_tensor, self.current_lid_tensor)
        normalized_lid = lid_outer / (lid_outer.max() + 1e-9)
        lid_multiplier = 1.0 + (self.lid_alpha * normalized_lid)
        return (base_loss * lid_multiplier).mean()


if __name__ == "__main__":
    BEST_CONFIGS = {
        "radoslaw-email": {"dim": 100, "lr": 0.01, "cap": None},
        "ia-hospital-ward-proximity-attr": {"dim": 100, "lr": 0.01, "cap": None},
        "ia-contacts_hypertext2009": {"dim": 100, "lr": 0.01, "cap": None},
        "ia-enron-employees": {"dim": 100, "lr": 0.01, "cap": None},
        "ia-primary-school-proximity-attr": {"dim": 100, "lr": 0.01, "cap": None},
        "fb-forum": {"dim": 200, "lr": 0.01, "cap": None},
        "email-Eu-core-temporal": {"dim": 200, "lr": 0.01, "cap": None},
        "CollegeMsg": {"dim": 50, "lr": 0.005, "cap": 20.0},
        "fb-messages": {"dim": 50, "lr": 0.005, "cap": 20.0},
    }

    EPOCHS = 300
    master_results = {}

    print("HEAD-TO-HEAD LINK PREDICTION: BEST MODELS")

    for DATASET, config in BEST_CONFIGS.items():
        print(f"DATASET: {DATASET}")
        print(
            f"[Config Loaded] Dim: {config['dim']} | LR: {config['lr']} | Cap: {config['cap']}"
        )

        try:
            try:
                databank = Databank()
                es = databank.get_data(DATASET)
            except:
                es, _ = load_event_sequence(DATASET)

            dg = DynamicGraph()

            res = (
                "month"
                if "email" in DATASET or "fb" in DATASET or "College" in DATASET
                else "day"
            )
            if "enron" in DATASET:
                res = "3-months"

            dg.init_from_event_sequence(es, time_resolution=res, directed=False)
            graph_sequence = dg.get_graph_sequence()

            if len(graph_sequence) < 2:
                print("Not enough snapshots for Link Prediction. Skipping.")
                continue

            print("\nPRE-COMPUTING LID SCORES")
            global_precomputed_lids = []
            for i, G_wrapper in enumerate(graph_sequence):
                G = G_wrapper.G
                if len(G.nodes) > 0:
                    lid_estimator = NCLIDEstimator(G, alpha=1.0)
                    lid_estimator.estimate_lids()
                    nodelist = list(G.nodes())
                    lid_scores = [lid_estimator.get_lid(n) for n in nodelist]
                    global_precomputed_lids.append(
                        torch.tensor(lid_scores, dtype=torch.float32)
                    )
                else:
                    global_precomputed_lids.append(
                        torch.tensor([], dtype=torch.float32)
                    )
            print("Pre-computation complete.")

            d, lr, cap = config["dim"], config["lr"], config["cap"]
            h = d * 2

            models_to_test = [
                (
                    "Distance_Basic",
                    CappedDistanceDynAE(
                        dim=d, hidden_dim=h, epochs=EPOCHS, lr=lr, pos_cap=cap
                    ),
                ),
                (
                    "Distance_Hub",
                    HubDistanceDynAE(
                        dim=d,
                        hidden_dim=h,
                        epochs=EPOCHS,
                        lr=lr,
                        pos_cap=cap,
                        hub_alpha=1.0,
                    ),
                ),
                (
                    "Distance_LID",
                    LIDDistanceDynAE(
                        dim=d,
                        hidden_dim=h,
                        epochs=EPOCHS,
                        lr=lr,
                        pos_cap=cap,
                        lid_alpha=1.0,
                        precomputed_lids=global_precomputed_lids,
                    ),
                ),
            ]

            dataset_results = {}
            distance_metric = EuclideanDistance()

            for model_name, ae in models_to_test:
                print(f"\nTESTING MODEL: {model_name}")
                start_t = time.time()
                ae.make_embedding(dg)

                link_predictor = UnsupervisedLinkPrediction(distance=distance_metric)
                evaluator = Predicter(linkpred=link_predictor, embedding=ae, dg=dg)
                evaluator.predict(print_results=False, take_top_k=False)

                valid_roc = [
                    res["roc_auc"]
                    for res in evaluator.results
                    if res.get("roc_auc") is not None
                ]
                valid_ap = [
                    res["average_precision"]
                    for res in evaluator.results
                    if res.get("average_precision") is not None
                ]

                mean_roc = np.mean(valid_roc) if valid_roc else 0.0
                mean_ap = np.mean(valid_ap) if valid_ap else 0.0

                dataset_results[model_name] = {"roc_auc": mean_roc, "avg_prec": mean_ap}
                print(
                    f"   -> ROC-AUC: {mean_roc:.4f} | Avg Prec: {mean_ap:.4f} | Time: {time.time() - start_t:.2f}s"
                )

            master_results[DATASET] = dataset_results

            print(f"FINAL LINK PREDICTION")

            print(f"{'Metric':<12} | {'Basic':<12} | {'Hub':<12} | {'LID':<12}")

            for metric, display_name in [
                ("roc_auc", "ROC-AUC"),
                ("avg_prec", "Avg Prec"),
            ]:
                row_str = f"{display_name:<12} | "
                scores = [dataset_results[name][metric] for name, _ in models_to_test]
                best_score = max(scores) if scores else 0
                for score in scores:
                    marker = "🎯" if score == best_score and score > 0 else "  "
                    row_str += f"{score:.4f} {marker}   | "
                print(row_str[:-3])

        except Exception as e:
            print(f"Failed to process dataset {DATASET}. Error: {e}")
