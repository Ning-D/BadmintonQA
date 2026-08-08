"""
Baseline selectors (VideoTree / BOLT / LVNet) on the full BadmintonQA set,
same two-phase protocol as run_aks_full.py.

Phase 1 (this script, backbone-agnostic, cached): one decode + BLIP-ITM +
CLIP-L pass per match, then for every question select frames at all budgets
(8/16/32/64) with the three methods:
  videotree : breadth/depth clustering (videotree_select.py, option-B repro)
              over CLIP-L feats + BLIP question relevance
  bolt      : inverse transform sampling over the BLIP question-relevance
              distribution (Liu et al. 2025)
  lvnet     : gpt-5.4-mini extracts visual keywords from the options, frames
              ranked by BLIP-ITM alignment to the keyword bag, top-N
Caches: badminton_data/vqa/table2/{sel}_cache_f{N}[.sIofK].json (same schema
as the AKS caches). LVNet keywords: table2/lvnet_keywords.json.

Phase 2 (answering) reuses run_aks_full.py:
  <env-python> run_aks_full.py --phase 2 --selector videotree --frames 32 --backbone qwen3vl

Usage:
  <qwen25vl-python> run_baselines_full.py --shard 0/2
"""
import os, sys, json, time, gc, argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "badminton_data")
SCR = os.path.dirname(os.path.abspath(__file__))
if SCR not in sys.path:
    sys.path.insert(0, SCR)

from run_table2 import load_qas  # noqa: E402

BUDGETS = [8, 16, 32, 64]
SELS = ["videotree", "bolt", "lvnet"]
KW_PATH = f"{ROOT}/vqa/table2/lvnet_keywords.json"
GPT_MODEL = "gpt-5.4-mini"   # the only OpenAI model we are allowed to use

SYSTEM_LVNET_KW = """\
You are given a multiple-choice video question with its options. Extract a BAG \
of atomic VISUAL keywords (concrete objects, activities, places, attributes) \
mentioned COLLECTIVELY across the options -- the things one would look for in \
the video to tell the options apart. Do NOT include the question's generic \
words or pick an answer. Return ONLY JSON: {"keywords":["...","..."]}"""


def cache_path(sel, n, suffix=""):
    return f"{ROOT}/vqa/table2/{sel}_cache_f{n}{suffix}.json"


def bolt_select(scores, fidx, n):
    """BOLT: inverse transform sampling over the relevance distribution --
    N evenly spaced quantiles of the score CDF, deduped, topped up by score."""
    s = np.asarray(scores, np.float64)
    rng = float(s.max() - s.min())
    p = np.ones_like(s) if rng < 1e-9 else (s - s.min()) / rng
    p = p / p.sum()
    cdf = np.cumsum(p)
    idx = {min(int(np.searchsorted(cdf, (j + 0.5) / n)), len(s) - 1)
           for j in range(n)}
    for i in np.argsort(-s):
        if len(idx) >= min(n, len(s)):
            break
        idx.add(int(i))
    return sorted(int(fidx[i]) for i in idx)


def _kw_call(q):
    from graph_clip import _gpt_json
    content = (q["question"] + "\nOptions:\n"
               + "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(q["options"])))
    try:
        r = _gpt_json(GPT_MODEL, SYSTEM_LVNET_KW, content, max_tokens=160) or {}
    except Exception as e:
        print(f"[baselines] !! keyword call failed for {q['id']}: {e}", flush=True)
        r = {}
    return [k.strip() for k in (r.get("keywords") or [])
            if isinstance(k, str) and k.strip()][:12]


def prefetch_keywords(questions, kwcache, workers=12):
    """Fill kwcache for all questions up front with a thread pool, so the
    per-match loop never blocks on the OpenAI API."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [q for q in questions if q["id"] not in kwcache]
    if not todo:
        return
    print(f"[baselines] prefetching keywords for {len(todo)} questions "
          f"({workers} workers)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for q, kw in zip(todo, ex.map(_kw_call, todo)):
            kwcache[q["id"]] = kw
    json.dump(kwcache, open(KW_PATH, "w"), ensure_ascii=False)
    print(f"[baselines] keywords done in {time.time()-t0:.0f}s", flush=True)


def get_keywords(q, kwcache):
    if q["id"] in kwcache:
        return kwcache[q["id"]]
    kwcache[q["id"]] = _kw_call(q)
    return kwcache[q["id"]]


def clip_feats(model, proc, frames, device, bs=64):
    import torch
    embs = []
    with torch.no_grad():
        for k in range(0, len(frames), bs):
            px = proc(images=frames[k:k + bs],
                      return_tensors="pt")["pixel_values"].to(device, torch.float16)
            f = model.get_image_features(pixel_values=px)
            embs.append(f.float().cpu().numpy())
    return np.concatenate(embs, 0).astype(np.float32)


def main():
    import torch
    from transformers import (BlipForImageTextRetrieval, AutoProcessor,
                              CLIPModel, CLIPProcessor)
    from aks_score import read_1fps, encode_vision, score_question, MODEL_ID as BLIP_ID
    from videotree_select import videotree_select

    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pool", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    sfx = f".s{si}of{sk}" if sk > 1 else ""

    qas = load_qas()
    by_video = defaultdict(list)
    for q in qas:
        by_video[q["video"]].append(q)

    import glob as _g
    caches = {}
    for sel in SELS:
        for n in BUDGETS:
            cp = cache_path(sel, n, sfx)
            cache = json.load(open(cp)) if os.path.exists(cp) else {}
            for other in _g.glob(cache_path(sel, n, "*").replace(".json", "") + "*.json"):
                if other == cp:
                    continue
                try:
                    for k, v in json.load(open(other)).items():
                        cache.setdefault(k, v)
                except Exception:
                    pass
            caches[(sel, n)] = (cp, cache)
    kwcache = json.load(open(KW_PATH)) if os.path.exists(KW_PATH) else {}

    todo = {v: [q for q in qs
                if any(q["id"] not in c for _, c in caches.values())]
            for v, qs in by_video.items()}
    todo = {v: qs for i, (v, qs) in enumerate(sorted(todo.items()))
            if qs and i % sk == si}
    n_todo = sum(len(qs) for qs in todo.values())
    print(f"[baselines] phase1 {SELS} f={BUDGETS}: {len(todo)} matches / "
          f"{n_todo} questions", flush=True)
    if not todo:
        return

    prefetch_keywords([q for qs in todo.values() for q in qs], kwcache)

    blip = BlipForImageTextRetrieval.from_pretrained(
        BLIP_ID, dtype=torch.float16).to(a.device).eval()
    bproc = AutoProcessor.from_pretrained(BLIP_ID)
    clipm = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", dtype=torch.float16).to(a.device).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    for mi, (v, qs) in enumerate(sorted(todo.items())):
        t0 = time.time()
        pool, fidx = read_1fps(v, max_pool=a.max_pool)
        if not pool:
            print(f"[baselines] !! no frames {v}", flush=True)
            continue
        ie = encode_vision(blip, bproc, pool, a.device)
        feats = clip_feats(clipm, clipp, pool, a.device)
        for q in qs:
            sc = np.asarray(score_question(blip, bproc, ie, q["question"],
                                           a.device), dtype=np.float32)
            kw = get_keywords(q, kwcache)
            kwtext = ", ".join(kw) if kw else q["question"]
            sck = np.asarray(score_question(blip, bproc, ie, kwtext,
                                            a.device), dtype=np.float32)
            for n in BUDGETS:
                picks = {
                    "videotree": lambda: [int(x) for x in
                                          videotree_select(feats, fidx, sc, n=n)],
                    "bolt": lambda: bolt_select(sc, fidx, n),
                    "lvnet": lambda: sorted(int(fidx[i])
                                            for i in np.argsort(-sck)[:n]),
                }
                for sel in SELS:
                    _, cache = caches[(sel, n)]
                    if q["id"] not in cache:
                        cache[q["id"]] = {"match": q["match_name"], "video": v,
                                          "picks": picks[sel]()}
        for cp, cache in caches.values():
            json.dump(cache, open(cp, "w"))
        json.dump(kwcache, open(KW_PATH, "w"), ensure_ascii=False)
        print(f"[baselines] [{mi+1}/{len(todo)}] {os.path.basename(v)[:44]:44} "
              f"{len(pool)} pool / {len(qs)} QA, {time.time()-t0:.0f}s", flush=True)
        del ie, feats
        gc.collect()
        torch.cuda.empty_cache()
    print("[baselines] phase1 complete", flush=True)


if __name__ == "__main__":
    main()
