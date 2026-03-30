import sys
import os
import networkx as nx
import numpy as np
import random
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.greval import GraphReconstructionEval
from tigra.core.loaders import load_event_sequence


try:
    from tigra.embeddings.sota.dynnode2vec import RWEmbEgoDelta, RandomWalkSampler
except ImportError:
    from tigra.embeddings.rndw.rndwbase import RWEmbEgoDelta, RandomWalkSampler


from tigra.evaluation.lideval import NCLIDEstimator


class LIDBanditSampler(RandomWalkSampler):
    """
    Multi-Armed Bandit that chooses between Structural (Common Neighbors)
    and LID-guided (Natural Community) exploration.
    """
    def __init__(self, graph: nx.Graph, nodes, lid_scores: dict, epsilon=0.1, alpha=0.1, num_walks=10, walk_length=32, num_workers=1):
        super().__init__(graph, nodes, num_workers=num_workers)
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.nw = num_walks
        self.wl = walk_length
        self.lid_scores = lid_scores 
        
        self.q_values = [0.5, 0.5] 
        self._neighbor_cache = {}

    def _get_neighbors_set(self, node):
        if node not in self._neighbor_cache:
            self._neighbor_cache[node] = set(self.G.neighbors(node))
        return self._neighbor_cache[node]

    # Structural Walk
    def _select_structural(self, current_node, neighbours):
        if len(neighbours) == 1: return neighbours[0]
        set_curr = self._get_neighbors_set(current_node)
        weights = [len(set_curr & self._get_neighbors_set(n)) + 0.1 for n in neighbours]
        total = sum(weights)
        return np.random.choice(neighbours, p=[w/total for w in weights])

    # LID-Guided Walk
    def _select_lid(self, neighbours):
        if len(neighbours) == 1: return neighbours[0]
        scores = np.array([self.lid_scores.get(n, 0.0) for n in neighbours])
        scores = scores - np.min(scores) + 0.1 
        probs = scores / np.sum(scores)
        return np.random.choice(neighbours, p=probs)

    def rnd_walks_from_node(self, start_node, walk_length):
        walk = [start_node]
        visited_in_walk = {start_node}
        curr_node = start_node
        
        while len(walk) < walk_length:
            neighbours = list(self.G.neighbors(curr_node))
            if not neighbours: break
            
            if random.random() < self.epsilon: 
                action = random.choice([0, 1])
            else: 
                action = np.argmax(self.q_values)

            if action == 0:
                next_node = self._select_structural(curr_node, neighbours)
            else:
                next_node = self._select_lid(neighbours)
            
            if next_node not in visited_in_walk:
                visited_in_walk.add(next_node)
                lid_bonus = self.lid_scores.get(next_node, 0.0)
                reward = 1.0 + (0.1 * lid_bonus) 
            else:
                reward = -0.1 
            
            self.q_values[action] += self.alpha * (reward - self.q_values[action])
            walk.append(next_node)
            curr_node = next_node
            
        return walk

    def select_next_node(self, s, c, p, n): pass
    def num_walks(self, node): return self.nw 
    def walk_length(self, node): return self.wl


class DynLIDBanditWalk(RWEmbEgoDelta):
    def __init__(self, dim, global_lid_scores, epsilon=0.1, alpha=0.1, num_walks=10, walk_length=32, **kwargs):
        super().__init__(dim, delta_nodes=True)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.epsilon = epsilon
        self.alpha = alpha
        self.global_lid_scores = global_lid_scores
        self.current_snap_idx = 0

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        lid_scores = self.global_lid_scores[self.current_snap_idx]
        self.current_snap_idx += 1
        
        return LIDBanditSampler(
            graph, nodes, 
            lid_scores=lid_scores,
            epsilon=self.epsilon,
            alpha=self.alpha,
            num_walks=self.num_walks, 
            walk_length=self.walk_length
        )


if __name__ == "__main__":
    DATASETS = [
        ("radoslaw-email", "month"),
        ("ia-hospital-ward-proximity-attr", "day"),
        ("ia-contacts_hypertext2009", "day"),
        ("ia-enron-employees", "3-months"),
        ("ia-primary-school-proximity-attr", "day"),
        ("fb-forum", "month"),
        ("email-Eu-core-temporal", "month"),
        ("CollegeMsg", "month"),
        ("fb-messages", "month")
    ]

    DIMS_TO_TEST = [10, 25, 50, 100, 200]
    
    CONFIGS = [
        {"name": "Eps=0.1, Alpha=0.1", "eps": 0.1, "alpha": 0.1},
        {"name": "Eps=0.3, Alpha=0.2", "eps": 0.3, "alpha": 0.2},
    ]

    print(f"LID-BANDIT GRID SEARCH: 2 Configs x {len(DIMS_TO_TEST)} Dims x {len(DATASETS)} Datasets")


    for DATASET, TIME_RES in DATASETS:
        print(f"PROCESSING DATASET: {DATASET} (Res: {TIME_RES})")

        
        try:
            try:
                databank = Databank()
                es = databank.get_data(DATASET)
            except:
                es, _ = load_event_sequence(DATASET)

            dg = DynamicGraph()
            dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
            graph_sequence = dg.get_graph_sequence()
            
            if len(graph_sequence) == 0: continue

            print("PRE-COMPUTING LID SCORES...")
            global_precomputed_lids = []
            
            for i, G_wrapper in enumerate(graph_sequence):
                G = G_wrapper.G
                if len(G.nodes) > 0:
                    print(f"    -> Snapshot {i} ({len(G.nodes)} nodes)")
                    lid_estimator = NCLIDEstimator(G, alpha=1.0)
                    lid_estimator.estimate_lids()
                    snap_lids = {n: lid_estimator.get_lid(n) for n in G.nodes()}
                    global_precomputed_lids.append(snap_lids)
                else:
                    global_precomputed_lids.append({})
            print("Pre-computation complete.\n")

            dataset_results = {cfg["name"]: {} for cfg in CONFIGS}

            for cfg in CONFIGS:
                print(f"Testing Config: {cfg['name']}")
                
                for d in DIMS_TO_TEST:
                    start_t = time.time()
                    rl_model = DynLIDBanditWalk(
                        dim=d, 
                        global_lid_scores=global_precomputed_lids,
                        epsilon=cfg["eps"],
                        alpha=cfg["alpha"]
                    )
                    
                    rl_model.make_embedding(dg)
                    
                    avg_f1 = []
                    for i, graph_wrapper in enumerate(graph_sequence):
                        emb_obj = rl_model.embeddings[i]
                        if emb_obj is None: continue
                        
                        for node, vec in emb_obj.embedding.items():
                            norm = np.linalg.norm(vec)
                            if norm > 0:
                                emb_obj.embedding[node] = vec / norm

                        gre = GraphReconstructionEval(graph_wrapper, emb_obj, report_warnings=False)
                        avg_f1.append(gre.getF1())

                    mean_f1 = np.mean(avg_f1) if avg_f1 else 0.0
                    dataset_results[cfg["name"]][d] = mean_f1
                    print(f"   -> Dim {d:<3} | F1: {mean_f1:.4f} | Time: {time.time() - start_t:.2f}s")
            

            print(f"BANDIT RESULTS FOR: {DATASET}")
            config_names = [cfg["name"] for cfg in CONFIGS]
            header = f"{'Dim':<5} | " + " | ".join([f"{name:<20}" for name in config_names])
            print(header)
            print("-" * len(header))
            
            for d in DIMS_TO_TEST:
                row_str = f"{d:<5} | "
                scores = [dataset_results[name].get(d, 0.0) for name in config_names]
                best_score = max(scores) if scores else 0
                
                for score in scores:
                    marker = "🎯" if score == best_score and score > 0 else "  "
                    row_str += f"{score:.4f} {marker}               | "
                print(row_str[:-3])

        except Exception as e:
            print(f"Failed to process dataset {DATASET}. Error: {e}")

    print("GLOBAL BANDIT BENCHMARK COMPLETE")