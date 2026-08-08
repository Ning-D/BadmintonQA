"""
AKS@N on the FULL BadmintonQA set (533 QAs incl. doubles), same protocol as
run_table2.py but with per-question AKS keyframe selection instead of uniform.

Phase 1 (backbone-agnostic, cached): decode each match at ~1fps (pool capped),
encode once with BLIP-ITM, score every question of that match, AKS-select N
frames per question. Seeds from the old tactic-only cache when params match.
Phase 2: a frozen VLM answers from each question's selected frames.

Usage:
  <qwen25vl-python> run_aks_full.py --phase 1                      # selection
  <env-python>      run_aks_full.py --phase 2 --backbone qwen3vl   # answering
Results: badminton_data/vqa/table2/aks_f{N}_{backbone}.json
"""
import os, sys, json, time, gc, argparse
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "badminton_data")
SCR = os.path.dirname(os.path.abspath(__file__))
if SCR not in sys.path:
    sys.path.insert(0, SCR)

from run_table2 import load_qas, prompt_of, parse_letter, load_backbone  # noqa: E402

OLD_CACHE = f"{ROOT}/vqa/aks_select_cache.json"      # tactic-only, f32/pool1500


def cache_path(frames, suffix=""):
    return f"{ROOT}/vqa/table2/aks_cache_f{frames}{suffix}.json"


def phase1(budgets, max_pool, device, shard="0/1"):
    import torch
    from transformers import BlipForImageTextRetrieval, AutoProcessor
    from aks_score import read_1fps, encode_vision, score_question, MODEL_ID as BLIP_ID
    from aks_select import aks_select

    qas = load_qas()
    by_video = defaultdict(list)
    for q in qas:
        by_video[q["video"]].append(q)

    si, sk = map(int, shard.split("/"))
    import glob as _g
    caches = {}   # budget -> (shard cache path, dict)
    for b in budgets:
        cp = cache_path(b, f".s{si}of{sk}" if sk > 1 else "")
        cache = json.load(open(cp)) if os.path.exists(cp) else {}
        # merge whatever other shards/main cache already selected
        for other in _g.glob(cache_path(b, "*").replace(".json", "") + "*.json"):
            if other == cp: continue
            try:
                for k, v in json.load(open(other)).items(): cache.setdefault(k, v)
            except Exception: pass
        # seed from the old tactic cache (same frames budget & pool)
        if b == 32 and os.path.exists(OLD_CACHE):
            old = json.load(open(OLD_CACHE))
            for k, v in old.items():
                cache.setdefault(k, v)
            print(f"[aks-full] seeded {len(old)} selections from tactic cache")
        caches[b] = (cp, cache)

    todo = {v: [q for q in qs
                if any(q["id"] not in c for _, c in caches.values())]
            for v, qs in by_video.items()}
    todo = {v: qs for i, (v, qs) in enumerate(sorted(todo.items()))
            if qs and i % sk == si}
    n_todo = sum(len(qs) for qs in todo.values())
    print(f"[aks-full] phase1 f={budgets}: {len(todo)} matches / {n_todo} "
          f"questions to select", flush=True)
    if not todo:
        return

    blip = BlipForImageTextRetrieval.from_pretrained(
        BLIP_ID, dtype=torch.float16).to(device).eval()
    bproc = AutoProcessor.from_pretrained(BLIP_ID)
    for mi, (v, qs) in enumerate(sorted(todo.items())):
        t0 = time.time()
        pool_frames, fidx = read_1fps(v, max_pool=max_pool)
        if not pool_frames:
            print(f"[aks-full] !! no frames {v}")
            continue
        ie = encode_vision(blip, bproc, pool_frames, device)
        for q in qs:
            sc = score_question(blip, bproc, ie, q["question"], device)
            for b, (_, cache) in caches.items():
                if q["id"] in cache:
                    continue
                picks = aks_select(sc, fidx, n=b)
                cache[q["id"]] = {"match": q["match_name"], "video": v,
                                  "picks": picks}
        for cp, cache in caches.values():
            json.dump(cache, open(cp, "w"))
        print(f"[aks-full] [{mi+1}/{len(todo)}] {os.path.basename(v)[:44]:44} "
              f"{len(pool_frames)} pool / {len(qs)} QA, {time.time()-t0:.0f}s",
              flush=True)
        del ie
        gc.collect()
        torch.cuda.empty_cache()
    print("[aks-full] phase1 complete")


def read_frames(video, idxs, max_side=512):
    cap = cv2.VideoCapture(video)
    out = []
    for i in sorted(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        h, w = bgr.shape[:2]
        s = max_side / max(h, w)
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def merged_cache(frames_budget, selector="aks"):
    import glob as _g
    cache = {}
    for p in sorted(_g.glob(f"{ROOT}/vqa/table2/{selector}_cache_f{frames_budget}*.json")):
        try:
            for k, v in json.load(open(p)).items(): cache.setdefault(k, v)
        except Exception: pass
    return cache


def phase2(frames_budget, backbone, device, selector="aks"):
    qas = load_qas()
    cache = merged_cache(frames_budget, selector)
    outp = f"{ROOT}/vqa/table2/{selector}_f{frames_budget}_{backbone}.json"
    rows = json.load(open(outp)) if os.path.exists(outp) else []
    done = {r["id"] for r in rows}
    todo = [q for q in qas if q["id"] not in done]
    print(f"[aks-full] phase2 {selector}/{backbone}: {len(todo)} QAs to answer "
          f"({len(done)} cached)", flush=True)
    if not todo:
        report(rows, outp)
        return
    vlm = load_backbone(backbone, device)
    t0 = time.time()
    for k, q in enumerate(todo):
        c = cache.get(q["id"])
        if not c:
            continue
        frames = read_frames(c["video"], c["picks"])
        try:
            raw = vlm.answer_mc(frames, prompt_of(q))
        except Exception as e:
            raw = f"ERROR:{e}"
        pred = parse_letter(raw)
        rows.append({"id": q["id"], "match": q["match_name"],
                     "capability": q["capability"], "type": q["type"],
                     "gold": q["answer"], "pred": pred,
                     "correct": pred == q["answer"], "raw": str(raw)[:80]})
        if (k + 1) % 25 == 0:
            json.dump(rows, open(outp, "w"))
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"[aks-full] {k+1}/{len(todo)} | running acc {acc:.3f} | "
                  f"{(time.time()-t0)/(k+1):.1f}s/QA", flush=True)
    report(rows, outp)


def report(rows, outp):
    n, nc = len(rows), sum(r["correct"] for r in rows)
    print(f"\n=== {os.path.basename(outp)}: {nc}/{n} = {nc/max(1,n)*100:.1f}% ===")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["capability"]][0] += r["correct"]
        by[r["capability"]][1] += 1
    for c, (a, b) in sorted(by.items()):
        print(f"  {c:22} {a:3}/{b:<3} = {a/b*100:.1f}%")
    json.dump(rows, open(outp, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2"], required=True)
    ap.add_argument("--frames", default="32",
                    help="frame budget; phase 1 accepts a comma list (8,16,64)")
    ap.add_argument("--max_pool", type=int, default=1500)
    ap.add_argument("--backbone", choices=["qwen3vl", "mplug", "llava"])
    ap.add_argument("--selector", default="aks",
                    choices=["aks", "videotree", "bolt", "lvnet",
                             "evicover", "evicoveru", "evicover_wosub"],
                    help="phase 2: which selector's frame cache to answer from")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    budgets = [int(x) for x in str(a.frames).split(",")]
    if a.phase == "1":
        phase1(budgets, a.max_pool, a.device, a.shard)
    else:
        assert len(budgets) == 1, "phase 2 takes a single --frames value"
        phase2(budgets[0], a.backbone, a.device, a.selector)
