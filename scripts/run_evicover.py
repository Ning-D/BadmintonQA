"""
EviCover (ours) phase 1 on the full BadmintonQA set — coverage-oriented frame
selection (paper Sec. 4 / METHOD.md), same two-phase protocol as the baselines.

Per match (question-agnostic, cached):
  1. decode the same 1 FPS pool as every other selector (max_pool=1500),
     embed with SigLIP;
  2. event segmentation: change-points on adjacent-frame cosine similarity,
     segments capped at TAU=30 s -> ~100-200 event nodes;
  3. one-sentence caption of each node's center frame (gpt-5.4-mini, 12
     workers, cached in evicover_nodes.json);
  4. node game index from tactics_inferred rally spans (singles; doubles
     nodes get game 0 = unknown, stratified by time instead).
Per question x budget N:
  EviCover   : gpt-5.4-mini reads question+options+node timeline, decomposes
               into sub-needs, picks K=N nodes under an equal per-sub-need
               quota; validated mechanically, quota shortfalls topped up by
               stratified-uniform unused nodes; parse failure -> EviCover-U.
  EviCover-U : LLM-free — stratified uniform nodes (equal shares per game,
               else per time-third).
  Frame per node: pool frame with max SigLIP similarity to the question.

Caches: table2/evicover_cache_f{N}[.sIofK].json, evicoveru_cache_f{N}...json
Phase 2 (answering) reuses run_aks_full.py:
  <env-python> run_aks_full.py --phase 2 --selector evicover --frames 32 --backbone qwen3vl

Usage: <qwen25vl-python> run_evicover.py [--shard 0/1] [--budgets 8,16,32,64]
"""
import os, sys, json, time, gc, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "badminton_data")
SCR = os.path.dirname(os.path.abspath(__file__))
if SCR not in sys.path:
    sys.path.insert(0, SCR)

from run_table2 import load_qas  # noqa: E402
from graph_clip import _gpt_json, _img  # noqa: E402

GPT_MODEL = "gpt-5.4-mini"   # the only OpenAI model allowed in this project
SIGLIP_ID = "google/siglip-so400m-patch14-384"
TAU_S = 30.0                 # max event-node length (seconds)
FPS = 30.0                   # broadcast fps (frame idx -> seconds)
NODES_PATH = f"{ROOT}/vqa/table2/evicover_nodes.json"
WORKERS = 12

SYSTEM_CAPTION = ("Describe this badminton broadcast frame in ONE short "
                  "sentence: what is shown (rally in play / serve / replay / "
                  "break / crowd / graphics) and anything notable. No preamble.")

# EviCover = caption-guided semantic node selection. The LLM reads the node
# captions and picks the K nodes most useful for the question; no sub-need
# decomposition and no coverage quota (an earlier variant used those and did
# strictly worse -- the coverage constraint hurt single-moment questions).
SYSTEM_SELECT = """\
You allocate a keyframe budget for a multiple-choice question about a full \
badminton match broadcast. You get the question, its four options, and a \
numbered timeline of event nodes (game index, start time, duration, caption).
Pick exactly K DISTINCT node indices that are most useful for telling the \
options apart. Prefer nodes whose captions indicate active play relevant to \
the question over replays/breaks/graphics.
Return ONLY JSON: {"picks": [<idx>, <idx>, ...]}"""

SYSTEM_SELECT_WOSUB = SYSTEM_SELECT   # back-compat alias for --variant wosub


def mmss(fr):
    s = int(fr / FPS)
    return f"{s//60}:{s%60:02d}"


# ---------------- phase A: nodes (segment + caption + game) ----------------

def _step_a(feats, min_len=2, max_len=30, z=1.0):
    """STEP A event segmentation:
    change-point segmentation on normalized frame features.
    Returns list of (start_idx, end_idx) inclusive."""
    T = feats.shape[0]
    if T <= min_len:
        return [(0, T - 1)]
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    d = 1.0 - (f[:-1] * f[1:]).sum(1)
    thr = d.mean() + z * d.std()
    bounds = [0]
    last = 0
    for i in range(1, T):
        cut = (i - last >= max_len) or (d[i - 1] > thr and i - last >= min_len)
        if cut:
            bounds.append(i)
            last = i
    segs = [(a, b - 1) for a, b in zip(bounds, bounds[1:])] + [(bounds[-1], T - 1)]
    return [(a, b) for a, b in segs if b >= a]


def segment_nodes(embs, fidx):
    """STEP A on the (possibly capped) pool grid: max_len is TAU_S expressed
    in grid steps, since the 1,500-frame pool can be coarser than 1 fps."""
    gap_s = float(np.median(np.diff(fidx))) / FPS if len(fidx) > 1 else 1.0
    max_len = max(2, int(round(TAU_S / max(gap_s, 1e-6))))
    segs = _step_a(embs, min_len=2, max_len=max_len, z=1.0)
    return [(a, b + 1) for a, b in segs]   # inclusive -> half-open [i0, i1)


def game_spans(match):
    p = f"{ROOT}/annotations/tactics_inferred/{match}.json"
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    d = d[0] if isinstance(d, list) else d
    spans = defaultdict(lambda: [1e18, -1])
    for rl in d.get("rallies", []):
        g = rl.get("game")
        s, e = rl.get("start_frame"), rl.get("end_frame")
        if g is None or s is None:
            continue
        spans[g][0] = min(spans[g][0], s)
        spans[g][1] = max(spans[g][1], e)
    return {g: tuple(v) for g, v in spans.items()}


def node_game(center_fidx, spans):
    for g, (s, e) in spans.items():
        if s <= center_fidx <= e:
            return g
    return 0


def caption_nodes(pool, nodes, cached):
    import cv2

    def one(k):
        i0, i1 = nodes[k]
        frame = pool[(i0 + i1) // 2]
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        try:
            r = _gpt_json(GPT_MODEL, SYSTEM_CAPTION,
                          [_img(bgr),
                           {"type": "text", "text": 'Return JSON: {"caption": "..."}'}],
                          max_tokens=60) or {}
            return k, str(r.get("caption", ""))[:140]
        except Exception as e:
            print(f"[evicover] !! caption fail node {k}: {e}", flush=True)
            return k, ""
    caps = list(cached) + [""] * max(0, len(nodes) - len(cached))
    caps = caps[:len(nodes)]
    todo = [k for k in range(len(nodes)) if not caps[k]]
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for k, c in ex.map(one, todo):
                caps[k] = c
    return caps


# ---------------- phase B: per-question selection ----------------

def stratified_nodes(meta, n):
    """EviCover-U: equal node shares per game (or per time-third if games
    unknown), evenly spaced within each stratum."""
    strata = defaultdict(list)
    if any(m["game"] for m in meta):
        for k, m in enumerate(meta):
            strata[m["game"] or 0].append(k)
    else:
        third = max(1, len(meta) // 3)
        for k in range(len(meta)):
            strata[k // third].append(k)
    keys = sorted(strata)
    base, extra = divmod(min(n, len(meta)), len(keys))
    out = []
    for j, g in enumerate(keys):
        take = base + (1 if j < extra else 0)
        arr = strata[g]
        if take >= len(arr):
            out += arr
        elif take > 0:
            idx = np.linspace(0, len(arr) - 1, take).astype(int)
            out += [arr[i] for i in idx]
    # top up if rounding lost slots
    for k in range(len(meta)):
        if len(out) >= min(n, len(meta)):
            break
        if k not in out:
            out.append(k)
    return sorted(out[:min(n, len(meta))])


def uniform_nodes(meta, n):
    """Time-uniform node choice (no game strata) — wosub top-up/fallback,
    so no coverage prior sneaks back into the ablation."""
    K = min(n, len(meta))
    return sorted(set(np.linspace(0, len(meta) - 1, K).astype(int).tolist()))


def select_wosub(q, meta, n):
    """Ablation: caption-guided LLM selection WITHOUT sub-need decomposition
    and WITHOUT the per-sub-need coverage quota."""
    lines = "\n".join(
        f"{k}: G{m['game'] or '?'} t={mmss(m['center'])} "
        f"dur={m['dur_s']:.0f}s — {m['caption'] or '(no caption)'}"
        for k, m in enumerate(meta))
    opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(q["options"]))
    content = (f"K = {min(n, len(meta))}\nQuestion: {q['question']}\n"
               f"Options:\n{opts}\n\nNode timeline:\n{lines}")
    try:
        r = _gpt_json(GPT_MODEL, SYSTEM_SELECT_WOSUB, content,
                      max_tokens=1000) or {}
        raw = r.get("picks", [])
        picks = [int(p["node"]) if isinstance(p, dict) else int(p)
                 for p in raw]
    except Exception:
        picks = []
    seen, valid = set(), []
    for k in picks:
        if 0 <= k < len(meta) and k not in seen:
            seen.add(k)
            valid.append(k)
    K = min(n, len(meta))
    if not valid:
        return uniform_nodes(meta, n)
    if len(valid) > K:
        valid = valid[:K]
    for k in uniform_nodes(meta, n):     # time-uniform top-up, no strata
        if len(valid) >= K:
            break
        if k not in seen:
            valid.append(k)
    return sorted(valid)


def select_evicover(q, meta, n):
    """EviCover node selection = caption-guided semantic choice (no sub-need
    decomposition, no coverage quota). Kept as a thin wrapper over the shared
    implementation so both the main path and --variant wosub are identical."""
    return select_wosub(q, meta, n)


def main():
    import torch
    from transformers import AutoModel, AutoProcessor
    from aks_score import read_1fps

    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--variant", default="full", choices=["full", "wosub"],
                    help="wosub = ablation: caption-guided LLM selection "
                         "without sub-need decomposition / coverage quota")
    ap.add_argument("--max_pool", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    si, sk = map(int, a.shard.split("/"))
    sfx = f".s{si}of{sk}" if sk > 1 else ""
    sels = ("evicover", "evicoveru") if a.variant == "full" \
        else ("evicover_wosub",)

    qas = load_qas()
    by_video = defaultdict(list)
    for q in qas:
        by_video[q["video"]].append(q)

    import glob as _g
    caches = {}
    for sel in sels:
        for n in budgets:
            cp = f"{ROOT}/vqa/table2/{sel}_cache_f{n}{sfx}.json"
            cache = json.load(open(cp)) if os.path.exists(cp) else {}
            for other in _g.glob(f"{ROOT}/vqa/table2/{sel}_cache_f{n}*.json"):
                if other == cp:
                    continue
                try:
                    for k, v in json.load(open(other)).items():
                        cache.setdefault(k, v)
                except Exception:
                    pass
            caches[(sel, n)] = (cp, cache)
    nodestore = json.load(open(NODES_PATH)) if os.path.exists(NODES_PATH) else {}

    todo = {v: [q for q in qs
                if any(q["id"] not in c for _, c in caches.values())]
            for v, qs in by_video.items()}
    todo = {v: qs for i, (v, qs) in enumerate(sorted(todo.items()))
            if qs and i % sk == si}
    print(f"[evicover] phase1 f={budgets}: {len(todo)} matches / "
          f"{sum(len(qs) for qs in todo.values())} questions", flush=True)
    if not todo:
        return

    sig = AutoModel.from_pretrained(SIGLIP_ID, dtype=torch.float16).to(a.device).eval()
    sproc = AutoProcessor.from_pretrained(SIGLIP_ID)

    @torch.no_grad()
    def img_feats(frames, bs=64):
        out = []
        for k in range(0, len(frames), bs):
            px = sproc(images=frames[k:k+bs], return_tensors="pt")["pixel_values"]
            f = sig.get_image_features(pixel_values=px.to(a.device, torch.float16))
            out.append(f.float().cpu().numpy())
        return np.concatenate(out, 0)

    @torch.no_grad()
    def txt_feat(text):
        tok = sproc(text=[text], return_tensors="pt", padding="max_length",
                    truncation=True)
        f = sig.get_text_features(input_ids=tok["input_ids"].to(a.device))
        return f.float().cpu().numpy()[0]

    for mi, (v, qs) in enumerate(sorted(todo.items())):
        t0 = time.time()
        match = qs[0]["match_name"]
        pool, fidx = read_1fps(v, max_pool=a.max_pool)
        if not pool:
            print(f"[evicover] !! no frames {v}", flush=True)
            continue
        fidx = np.asarray(fidx)
        embs = img_feats(pool)

        nodes = segment_nodes(embs, fidx)
        spans = game_spans(match)
        st = nodestore.get(match, {})
        caps = caption_nodes(pool, nodes, st.get("captions", []))
        meta = [{"i0": i0, "i1": i1,
                 "center": int(fidx[(i0+i1)//2]),
                 "dur_s": float((fidx[i1-1]-fidx[i0])/FPS + 1),
                 "game": node_game(int(fidx[(i0+i1)//2]), spans),
                 "caption": caps[k]}
                for k, (i0, i1) in enumerate(nodes)]
        nodestore[match] = {"n_nodes": len(nodes), "captions": caps}
        json.dump(nodestore, open(NODES_PATH, "w"), ensure_ascii=False)

        en = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

        def frame_of_node(k, qfeat):
            i0, i1 = nodes[k]
            j = i0 + int(np.argmax(en[i0:i1] @ qfeat))
            return int(fidx[j])

        def one_question(q):
            qf = txt_feat(q["question"])
            qf = qf / (np.linalg.norm(qf) + 1e-8)
            out = {}
            for n in budgets:
                if a.variant == "full":
                    _, uc = caches[("evicoveru", n)]
                    if q["id"] not in uc:
                        ks = stratified_nodes(meta, n)
                        out[("evicoveru", n)] = sorted(frame_of_node(k, qf)
                                                       for k in ks)
                    _, ec = caches[("evicover", n)]
                    if q["id"] not in ec:
                        ks, fell_back = select_evicover(q, meta, n)
                        out[("evicover", n)] = sorted(frame_of_node(k, qf)
                                                      for k in ks)
                else:
                    _, wc = caches[("evicover_wosub", n)]
                    if q["id"] not in wc:
                        ks = select_wosub(q, meta, n)
                        out[("evicover_wosub", n)] = sorted(frame_of_node(k, qf)
                                                            for k in ks)
            return q, out

        # SigLIP text feats + LLM selection are thread-safe; GPU calls are
        # serialized by the GIL around each torch op, fine at this scale.
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for q, out in ex.map(one_question, qs):
                for key, picks in out.items():
                    caches[key][1][q["id"]] = {"match": q["match_name"],
                                               "video": v, "picks": picks}
        for cp, cache in caches.values():
            json.dump(cache, open(cp, "w"))
        print(f"[evicover] [{mi+1}/{len(todo)}] {os.path.basename(v)[:44]:44} "
              f"{len(nodes)} nodes / {len(qs)} QA, {time.time()-t0:.0f}s", flush=True)
        del embs, en
        gc.collect()
        torch.cuda.empty_cache()
    print("[evicover] phase1 complete", flush=True)


if __name__ == "__main__":
    main()
