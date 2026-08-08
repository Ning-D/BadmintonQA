"""
VideoTree keyframe selection (option-B adaptation of Ziyang412/VideoTree).

Keeps VideoTree's structure -- breadth (KMeans clustering of frame features for
visual COVERAGE) + depth (allocate MORE frames to query-RELEVANT clusters via
hierarchical sub-clustering) -- but, instead of EVA-CLIP-8B features + LaViLa
captions + GPT relevance, it uses light CLIP-L frame features (videotree_features.py)
and the per-frame BLIP-ITM relevance we already compute for AKS (cache_aks). The
output is a budget-capped set of N keyframe indices fed to Qwen2-VL / mPLUG-Owl3.
"""
import numpy as np
from sklearn.cluster import KMeans


def _l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _subcluster_reps(feats, gframes, k):
    """Split a cluster into k sub-clusters (cosine KMeans) -> representative
    (closest to centroid) global frame index per sub-cluster. = VideoTree depth."""
    k = int(min(k, len(gframes)))
    if k <= 1:
        # representative of the whole cluster
        Xn = _l2n(feats)
        c = Xn.mean(0, keepdims=True)
        return [gframes[int(np.linalg.norm(Xn - c, axis=1).argmin())]]
    Xn = _l2n(feats)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(Xn)
    reps = []
    for c in range(k):
        m = np.where(km.labels_ == c)[0]
        if len(m) == 0:
            continue
        d = np.linalg.norm(Xn[m] - km.cluster_centers_[c], axis=1)
        reps.append(gframes[int(m[d.argmin()])])
    return reps


def videotree_select(feats, frames, scores, n=64, init_clusters=8):
    """feats [F,D] CLIP feats; frames [F] frame indices (1fps); scores [F] BLIP
    relevance. Returns up to n selected frame indices (temporal order)."""
    F = len(frames)
    feats = np.asarray(feats, np.float32)
    scores = np.asarray(scores, np.float32)
    frames = list(frames)
    if F <= n:
        return sorted(frames)

    # ---- breadth: cluster all frames for visual coverage ----
    C = int(min(init_clusters, F))
    km = KMeans(n_clusters=C, n_init=4, random_state=0).fit(_l2n(feats))
    labels = km.labels_

    # ---- relevance per cluster (mean BLIP-ITM score) ----
    rel = np.array([scores[labels == c].mean() if (labels == c).any() else 0.0
                    for c in range(C)], np.float64)
    rel = np.clip(rel, 1e-3, None)

    # ---- depth budget: 1/cluster (breadth) + remainder weighted by relevance ----
    alloc = np.ones(C, int)
    extra = n - C
    if extra > 0:
        add = np.floor(rel / rel.sum() * extra).astype(int)
        alloc += add
        for c in np.argsort(-rel)[:extra - int(add.sum())]:
            alloc[c] += 1

    # ---- within each cluster: sub-cluster into alloc[c] representatives ----
    out = []
    for c in range(C):
        m = np.where(labels == c)[0]
        if len(m) == 0:
            continue
        out.extend(_subcluster_reps(feats[m], [frames[i] for i in m], alloc[c]))

    out = sorted(set(out))
    if len(out) > n:                       # enforce hard budget by relevance
        sc = {frames[i]: scores[i] for i in range(F)}
        out = sorted(sorted(out, key=lambda f: -sc.get(f, 0))[:n])
    return out
