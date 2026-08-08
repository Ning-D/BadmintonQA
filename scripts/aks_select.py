"""
AKS adaptive keyframe selection.

`meanstd` is copied verbatim from ncTimTang/AKS (frame_select.py); `aks_select`
wraps it the same way AKS's main() does (normalize -> meanstd -> depth-weighted
top-k per leaf) and then enforces a hard budget of N frames so the comparison
against our policy-graph runs uses the SAME frame budget.

Input:  per-frame relevance scores + the corresponding video frame indices
        (1 fps, as in AKS feature_extract.py).
Output: sorted list of selected video frame indices (<= n).
"""
import heapq
import numpy as np


def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[:len(score) // 2]
            score2 = score[len(score) // 2:]
            fn1 = fn[:len(score) // 2]
            fn2 = fn[len(score) // 2:]
            split_scores.append(dict(score=score1, depth=depth + 1))
            split_scores.append(dict(score=score2, depth=depth + 1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(
            len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn
    return all_split_score, all_split_fn


def aks_select(scores, frames, n=64, t1=0.8, t2=-100, all_depth=5):
    """scores, frames: parallel lists (frames = video frame indices @1fps).
    Returns up to n selected frame indices (sorted by time)."""
    scores = list(scores)
    frames = list(frames)
    if len(scores) <= n:
        return sorted(frames)
    arr = np.asarray(scores, dtype=np.float64)
    rng = float(arr.max() - arr.min())
    norm = (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)
    a, b = meanstd(len(norm), [dict(score=norm, depth=0)], n, [list(frames)],
                   t1, t2, all_depth)
    pairs = []  # (frame_idx, normalized_score) over leaf segments (disjoint)
    for s, f in zip(a, b):
        sc = s['score']
        f_num = max(1, int(n / 2 ** (s['depth'])))
        topk = heapq.nlargest(f_num, range(len(sc)), sc.__getitem__)
        pairs.extend((f[t], float(sc[t])) for t in topk)
    # hard budget: keep the n highest-scoring, then restore temporal order
    if len(pairs) > n:
        pairs.sort(key=lambda p: -p[1])
        pairs = pairs[:n]
    return sorted({fr for fr, _ in pairs})
