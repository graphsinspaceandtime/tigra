import sys, os

sys.path.append("..")
sys.path.append(os.path.abspath("."))

import tigra, networkx as nx
from tigra.embeddings.embedding import FSEmbedding

import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from incdbscan import IncrementalDBSCAN
from networkx.algorithms.community import modularity
from tigra.core.dgraph import DynamicGraph
from tigra.core.loaders import load_event_sequence
from sklearn.cluster import KMeans



def _patch_number_of_edges(cls):
    if not hasattr(cls, "number_of_edges") and hasattr(cls, "size"):
        cls.number_of_edges = cls.size
    if not hasattr(cls, "number_of_edges") and hasattr(cls, "num_edges"):
        cls.number_of_edges = cls.num_edges


_patch_number_of_edges(nx.Graph)
_patch_number_of_edges(tigra.core.dgraph.Graph)
_patch_number_of_edges(nx.DiGraph)


def to_normed_dict(d):
    out = {}
    for k, v in d.items():
        x = np.asarray(v, dtype=float)
        n = np.linalg.norm(x)
        if np.all(np.isfinite(x)) and n != 0.0:
            out[k] = x / n
    return out

def gen_embedding_matrix(node_embeddings, node_ids):
    rows = []
    for node_id in node_ids:
        rows.append(node_embeddings[node_id])
    return np.array(rows)


def silhouette(X, labels):
    m = labels >= 0
    labs = labels[m]
    if np.sum(m) < 2 or len(np.unique(labs)) < 2:
        return np.nan
    try:
        return silhouette_score(X[m], labs)
    except Exception:
        return np.nan


def nmi(prev_labels, curr_labels):
    if not prev_labels:
        return np.nan
    common = list(set(prev_labels) & set(curr_labels))
    if len(common) < 2:
        return np.nan
    a = np.array([prev_labels[n] for n in common])
    b = np.array([curr_labels[n] for n in common])
    
    valid = (a >= 0) & (b >= 0) & np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return normalized_mutual_info_score(a, b)
    except ValueError:
        return np.nan


def update_model(clusterer, model_nodes, model_vecs, curr_normed):
    curr_nodes = set(curr_normed.keys())
    new_nodes = curr_nodes - model_nodes
    if new_nodes:
        X_new = np.array([curr_normed[n] for n in new_nodes])
        clusterer.insert(X_new)
        for n in new_nodes:
            model_nodes.add(n)
            model_vecs[n] = curr_normed[n]
    removed = model_nodes - curr_nodes
    if removed:
        X_rm = np.array([model_vecs[n] for n in removed])
        clusterer.delete(X_rm)
        for n in removed:
            model_nodes.remove(n)
            model_vecs.pop(n, None)
    return model_nodes, model_vecs


def labels_for_snapshot(clusterer, model_nodes, model_vecs, curr_normed):
    curr_nodes = set(curr_normed.keys())
    query_nodes = sorted(curr_nodes & model_nodes)
    
    if not query_nodes:
        empty_labels = {n: -1 for n in curr_nodes}
        return empty_labels, np.empty((0, 0)), np.array([]), query_nodes
    
    mat = gen_embedding_matrix(model_vecs, query_nodes)
    cluster_labels = clusterer.get_cluster_labels(mat)
    
    out_labels = {n: -1 for n in curr_nodes}
    
    for i in range(len(query_nodes)):
        out_labels[query_nodes[i]] = int(cluster_labels[i])
    
    return out_labels, mat, cluster_labels, query_nodes


def build_eps_range_from_kdist(X_normed, k, knee="middle", width=12, show_plot=True):
    if X_normed.size == 0:
        raise ValueError("No data for eps.")
    
    valid = []
    for row in X_normed:
        if np.all(np.isfinite(row)):
            valid.append(row)
    X = np.array(valid)
    
    if len(X) < k + 1:
        raise ValueError("Not enough data for eps calculation")
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    kd = np.sort(dists[:, k - 1])
    
    p05, p20, p40, p60, p80, p95 = np.percentile(kd, [5, 20, 40, 60, 80, 95])
    if knee == "lower":
        lo, hi = p20, p60
    elif knee == "upper":
        lo, hi = p60, p95
    else:
        lo, hi = p40, p80
    return np.linspace(lo, hi, width)

def cluster_kmeans(embeddings_list, target_k, graph_sequence):
    metrics = {"nmi": [], "silhouette": [], "n_clusters": [], "q": []}
    prev_labels = None

    for i, emb in enumerate(embeddings_list):
        cur = to_normed_dict(emb.embedding)
        if not cur:
            continue
        nodes = sorted(cur.keys())
        X = np.array([cur[n] for n in nodes])

        k = target_k

        if isinstance(target_k, list):
            k = target_k[i]

        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        
        lab_dict = {}
        for i in range(len(nodes)):
            lab_dict[nodes[i]] = int(labels[i])

        sil = silhouette(X, labels)
        nmi = nmi(prev_labels, lab_dict)
        g = graph_sequence[i].G
        comm_sets = {}
        for node, comm in lab_dict.items():
            if node in g:
                comm_sets.setdefault(comm, set()).add(node)
        comm_list = list(comm_sets.values())
        missing = (
            set(g.nodes()) - set().union(*comm_list) if comm_list else set(g.nodes())
        )
        for n in missing:
            comm_list.append({n})
        q = modularity(g, comm_list, weight="weight")

        metrics["n_clusters"].append(k)
        metrics["silhouette"].append(sil)
        metrics["nmi"].append(nmi if not np.isnan(nmi) else None)
        metrics["q"].append(q)

        prev_labels = lab_dict

    return metrics


def grid_search_params(
    embeddings_list,
    k_for_kdist=10,
    knee="middle",
    min_pts_range=None,
    n_snapshots_test=5,
    target_k=None,
    k_window=(0.5, 2.0),
):
    if min_pts_range is None:
        min_pts_range = [5, 8, 10, 15]

    snaps = embeddings_list[:n_snapshots_test]
    first = to_normed_dict(snaps[0].embedding)
    
    if first:
        X0 = np.array(list(first.values()))
    else:
        X0 = np.empty((0, 0))
        
    eps_range = build_eps_range_from_kdist(
        X0, k=k_for_kdist, knee=knee, width=12, show_plot=True
    )

    print("GRID SEARCH")
    print(f" min_pts: {min_pts_range}, eps: {np.round(eps_range, 3)}\n")

    rows = []
    for eps in eps_range:
        for min_pts in min_pts_range:
            clusterer = IncrementalDBSCAN(eps=float(eps), min_pts=int(min_pts))
            model_nodes, model_vecs = set(), {}
            sils, ncs, noises = [], [], []
            for emb in snaps:
                cur = to_normed_dict(emb.embedding)
                model_nodes, model_vecs = update_model(
                    clusterer, model_nodes, model_vecs, cur
                )
                lab_dict, Xq, Lq, qn = labels_for_snapshot(
                    clusterer, model_nodes, model_vecs, cur
                )
                if len(qn) == 0:
                    continue
                nc = len(set(Lq)) - (1 if -1 in Lq else 0)
                nn = int(np.sum(Lq == -1))
                ncs.append(nc)
                noises.append(nn / len(Lq))
                sils.append(silhouette(Xq, Lq))

            avg_sil = np.nanmean(sils) if sils else -999
            avg_nc = np.nanmean(ncs) if ncs else np.nan
            avg_noise = np.nanmean(noises) if noises else np.nan
            rows.append(
                {
                    "eps": float(eps),
                    "min_pts": int(min_pts),
                    "avg_silhouette": float(avg_sil),
                    "avg_clusters": float(avg_nc),
                    "avg_noise_ratio": float(avg_noise),
                }
            )
            print(
                f"eps={eps:.3f}, min_pts={min_pts} | sil={avg_sil:.4f}, clusters={avg_nc:.1f}, noise={avg_noise:.1%}"
            )

    df = pd.DataFrame(rows)
    if target_k is not None and np.isfinite(df["avg_clusters"]).any():
        lo = target_k * k_window[0]
        hi = target_k * k_window[1]
        cand = df[(df["avg_clusters"] >= lo) & (df["avg_clusters"] <= hi)]
        if len(cand) > 0:
            df_sorted = cand.sort_values(
                ["avg_silhouette", "avg_clusters"], ascending=[False, True]
            )
        else:
            df_sorted = df.sort_values("avg_silhouette", ascending=False)
    else:
        df_sorted = df.sort_values("avg_silhouette", ascending=False)

    best = df_sorted.iloc[0]
    print("best params")
    print(f"eps={best['eps']:.3f}, min_pts={int(best['min_pts'])}")
    print(
        f"Avg silhouette={best['avg_silhouette']:.4f}, Avg clusters={best['avg_clusters']:.1f}, Avg noise={best['avg_noise_ratio']:.1%}"
    )
    return best["eps"], int(best["min_pts"]), df_sorted


def cluster_dynamic_embeddings(embeddings_list, eps, min_pts):
    clusterer = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
    model_nodes, model_vecs = set(), {}
    all_labels = []
    metrics = {"nmi": [], "silhouette": [], "n_clusters": [], "n_noise": []}
    prev_labels = None

    for t, emb in enumerate(embeddings_list, start=1):
        print(f"\nSnapshot {t}")
        cur = to_normed_dict(emb.embedding)
        model_nodes, model_vecs = update_model(clusterer, model_nodes, model_vecs, cur)
        lab_dict, Xq, Lq, qn = labels_for_snapshot(
            clusterer, model_nodes, model_vecs, cur
        )
        all_labels.append(lab_dict)
        if len(qn) == 0:
            print("empty snapshot, skip")
            prev_labels = None
            continue

        nc = len(set(Lq)) - (1 if -1 in Lq else 0)
        nn = int(np.sum(Lq == -1))
        sil = silhouette(Xq, Lq)
        nmi = nmi(prev_labels, lab_dict)

        metrics["n_clusters"].append(nc)
        metrics["n_noise"].append(nn)
        metrics["silhouette"].append(sil)
        if not np.isnan(nmi):
            metrics["nmi"].append(nmi)

        print(f"cl={nc}, noise={nn}, NMI={nmi:.4f}, sil={sil:.4f}")

        prev_labels = lab_dict
    return all_labels, metrics

def print_summary(metrics):
    print("\nSUMMARY")
    nmi_vals = [v for v in metrics["nmi"] if v is not None]
    if nmi_vals:
        print(f"Avg NMI: {np.mean(nmi_vals):.4f}")
    if metrics["silhouette"]:
        print(f"Avg Silhouette: {np.nanmean(metrics['silhouette']):.4f}")
    print(f"Avg clusters: {np.mean(metrics['n_clusters']):.2f}")


def load_novel_emb(dataset):
    es, resol = load_event_sequence(dataset)
    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=resol, directed=False)
    dg.info()
    return dg


names = [
    "CollegeMsg",
    "email-Eu-core-temporal",
    "fb-forum",
    "fb-messages",
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    "ia-primary-school-proximity-attr",
    # "ia-reality-call",
    "radoslaw-email",
]

if __name__ == "__main__":
    # k_for_kdist = 12
    # k_window = (0.8, 1.2)

    emb_algorithms = {
        "deep_lid": "emb/lid/deep_lid",
        "deep_hub": "emb/hub/deep_hub",
        "dn2v": "emb/dn2v",
    }

    for emb_name, emb_path in emb_algorithms.items():
        print(f"EMBEDDING: {emb_name}")

        for dataset in names:
            print(dataset)

            louvain_df = pd.read_csv(
                f"community_detection/results_{dataset}.csv",
                na_values=["None"],
            )
            target_k = louvain_df["NC"].tolist()
            print(f"Target k from Louvain: {target_k}")

            dg = load_novel_emb(dataset)
            graph_sequence = dg.get_graph_sequence()

            emb = FSEmbedding(f"{emb_path}/{dataset}")

            # best_eps, best_min_pts, _ = grid_search_params(
            #     embeddings_list=emb.embeddings,
            #     k_for_kdist=k_for_kdist,
            #     knee="middle",
            #     min_pts_range=[7, 10, 12, 15],
            #     n_snapshots_test=5,
            #     target_k=target_k,
            #     k_window=k_window,
            # )

            # print("best params")
            # labels, metrics = cluster_dynamic_embeddings(
            #     emb.embeddings, eps=best_eps, min_pts=best_min_pts
            # )
            metrics = cluster_kmeans(
                emb.embeddings, target_k=target_k, graph_sequence=graph_sequence
            )
            
            print_summary(metrics)

            os.makedirs("clustering/{emb_name}", exist_ok=True)
            with open(f"clustering/{emb_name}/{dataset}.csv", "w") as f:
                f.write("snapshot,NC,NMI,SIL,Q\n")
                for t in range(len(metrics["n_clusters"])):
                    f.write(
                        f"{t},{metrics['n_clusters'][t]},{metrics['nmi'][t]},{metrics['silhouette'][t]},{metrics['q'][t]}\n"
                    )
