"""
Static clip-level evidence graph, adapted from belief-reversal (H1/H2) to
plain multiple-choice QA, with the DYNAMIC part removed.

Pipeline (one pass over the whole video, NO cross-step state update):
  1. Cut the video into fixed-interval clips (default 30s)
  2. Batch-score clips with the policy LLM (gpt-5.4-mini, 10 clips/call,
     1 mid-frame each) -> relevance 1-5 + short observation
  3. Pick top-k seeds (score>=3)
  4. BFS edge discovery from seeds over high-scoring candidates: the policy
     LLM types each clip pair (supports/causes/precedes/contradicts), keep
     edges with confidence>=thresh, grow graph up to max_nodes
  5. Greedy minimal sufficient chain (ask the LLM if the chain can answer yet)
  6. Answer: spread a fixed frame budget (default 16, == uniform baseline)
     across the chain clips in temporal order + their observations -> GPT-4o

Removed vs original (the only "dynamic" parts): check_state_update() and the
belief-reversal edge types (challenges/reinterprets/setup_for).
"""
import base64
import itertools
import json
import os
import re
import threading
import time
from collections import deque

import cv2
import numpy as np
from openai import OpenAI

def _make_clients():
    """Pool of OpenAI clients. Set OPENAI_API_KEYS (comma-separated) to round-robin
    over multiple keys -> higher throughput / dodges per-key rate limits. Falls back
    to a single default client (reads OPENAI_API_KEY) when OPENAI_API_KEYS is unset."""
    keys = [k.strip() for k in os.environ.get("OPENAI_API_KEYS", "").split(",") if k.strip()]
    return [OpenAI(api_key=k) for k in keys] if keys else [OpenAI()]


_clients = _make_clients()
_client = _clients[0]                        # back-compat: first key in the pool
_client_cycle = itertools.cycle(_clients)
_cycle_lock = threading.Lock()


def _next_client():
    with _cycle_lock:
        return next(_client_cycle)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "outputs", "cache_clipgraph")

EDGE_RELATIONS = ["supports", "causes", "precedes", "contradicts", "unrelated"]


# ── frame I/O (OpenCV; Video-MME has no pre-extracted frames) ────────────────
def _open(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if total > 0 else 0.0
    return cap, fps, dur


def _grab(cap, fps, sec, max_side):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    return rgb


def _b64(frame, quality=80):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def _img(frame):
    return {"type": "image_url", "image_url": {
        "url": f"data:image/jpeg;base64,{_b64(frame)}", "detail": "low"}}


def _hms(sec):
    sec = int(sec)
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


_COST = {"calls": 0, "in": 0, "out": 0}  # token accounting (GPT_COST_LOG)


def _gpt_json(model, system, content, max_tokens=800):
    for attempt in range(3):
        try:
            r = _next_client().chat.completions.create(
                model=model, max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": content}])
            u = getattr(r, "usage", None)
            if u is not None and os.environ.get("GPT_COST_LOG"):
                _COST["calls"] += 1
                _COST["in"] += getattr(u, "prompt_tokens", 0) or 0
                _COST["out"] += getattr(u, "completion_tokens", 0) or 0
                with open(os.environ["GPT_COST_LOG"], "w") as f:
                    json.dump(_COST, f)
            text = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group()) if m else None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2)


# ── prompts (multiple-choice adaptation) ────────────────────────
SYSTEM_SCORE = """\
You are scoring short video clips for how useful they are for answering a \
multiple-choice question about a long video.
For each clip, output a relevance score 1-5:
  1 = completely unrelated to the question
  2 = background / weak contextual link
  3 = somewhat relevant, minor evidence
  4 = relevant, clear evidence or context
  5 = highly relevant, key evidence for answering the question
Also give a brief observation (<=80 chars) of what the clip shows.
Output ONLY valid JSON:
{"clips": [{"clip_idx": 0, "score": 1, "observation": "..."}, ...]}"""

SYSTEM_EDGE = """\
You are analyzing two video clips to determine their relationship regarding a \
specific multiple-choice question.
Classify the relationship of Clip B to Clip A:
  supports    : B adds evidence in the same direction as A for the question
  causes      : A causes / leads to what is seen in B
  precedes    : A happens before B and is part of the same event chain
  contradicts : B conflicts with A's implication for the question
  unrelated   : no meaningful link regarding the question
Output ONLY JSON:
{"relation": "...", "direction": "A_to_B|B_to_A|bidirectional", \
"reason": "<=100 chars", "confidence": 1-5}"""

SYSTEM_SUFFICIENCY = """\
You are selecting a minimum sufficient evidence chain to answer a \
multiple-choice question.
Return JSON:
{"sufficient": true or false, "confidence": 0.0-1.0,
 "missing_info": "what is still needed (or 'none')",
 "best_candidate_id": "node_id or null",
 "why_needed": "what gap this fills"}
Output JSON only."""


def _qtext(item):
    opts = "\n".join(item["options"])
    return f"{item['question']}\nOptions:\n{opts}"


# ── pipeline steps ───────────────────────────────────────────────────────────
def _make_clips(dur, interval):
    clips, t = [], 0.0
    while t < dur - 1.0:
        clips.append((t, min(t + interval, dur)))
        t += interval
    return clips


def _score_clips(cap, fps, clips, qtext, batch, max_side, model):
    # 1. extract mid frames + build content per batch (serial: cv2 cap)
    payloads = []
    for s in range(0, len(clips), batch):
        chunk = clips[s:s + batch]
        content = [{"type": "text", "text":
                    f"Question: {qtext}\n\nRate each of the {len(chunk)} "
                    f"clips below for relevance to the question:"}]
        for i, (t0, t1) in enumerate(chunk):
            content.append({"type": "text",
                            "text": f"\n--- Clip {i} [{_hms(t0)} - {_hms(t1)}] ---"})
            fr = _grab(cap, fps, (t0 + t1) / 2, max_side)
            content.append(_img(fr) if fr is not None
                           else {"type": "text", "text": "(no frame)"})
        payloads.append((chunk, content))

    # 2. score all batches CONCURRENTLY (the slow part is the API, not the GPU)
    def _do(payload):
        chunk, content = payload
        obj = _gpt_json(model, SYSTEM_SCORE, content, max_tokens=900)
        results = (obj or {}).get("clips", []) if isinstance(obj, dict) else []
        out = [{"score": 1, "observation": ""} for _ in chunk]
        for r in results:
            try:
                idx = int(r.get("clip_idx", 0))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(out):
                out[idx] = {"score": int(r.get("score", 1)),
                            "observation": str(r.get("observation", ""))}
        return [{"t0": t0, "t1": t1, "score": o["score"],
                 "observation": o["observation"]}
                for (t0, t1), o in zip(chunk, out)]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(payloads) or 1)) as ex:
        per = list(ex.map(_do, payloads))
    return [s for batch_scores in per for s in batch_scores]


def _clip_frames(cap, fps, t0, t1, n, max_side):
    if n <= 1:
        tss = [(t0 + t1) / 2]
    else:
        m = (t1 - t0) * 0.1
        tss = np.linspace(t0 + m, t1 - m, n).tolist()
    out = []
    for ts in tss:
        fr = _grab(cap, fps, ts, max_side)
        if fr is not None:
            out.append(fr)
    return out


def _get_edge(cap, fps, A, B, qtext, max_side, model):
    content = [{"type": "text", "text":
                f"Question: {qtext}\n\n=== Clip A [{_hms(A['t0'])} - "
                f"{_hms(A['t1'])}] ===\nObservation: {A['observation']}"}]
    content += [_img(f) for f in A["frames"]]
    content += [{"type": "text", "text":
                 f"\n=== Clip B [{_hms(B['t0'])} - {_hms(B['t1'])}] ===\n"
                 f"Observation: {B['observation']}"}]
    content += [_img(f) for f in B["frames"]]
    content.append({"type": "text",
                    "text": "\nClassify the relationship of B to A."})
    obj = _gpt_json(model, SYSTEM_EDGE, content, max_tokens=200) or {}
    rel = obj.get("relation", "unrelated")
    return {"relation": rel if rel in EDGE_RELATIONS else "unrelated",
            "direction": obj.get("direction", "A_to_B"),
            "reason": obj.get("reason", ""),
            "confidence": int(obj.get("confidence", 0) or 0)}


def _min_chain(nodes_list, qtext, max_chain, model):
    """Greedy minimal sufficient chain over graph nodes (by score)."""
    cands = sorted(nodes_list, key=lambda x: -x["score"])
    if not cands:
        return []
    chain = [cands[0]]
    remaining = cands[1:]
    for _ in range(max_chain - 1):
        if not remaining:
            break
        chain_txt = "\n".join(f"[{n['id']} @ {n['t0']:.0f}s] "
                              f"{n['observation'][:70]}" for n in chain)
        rem_txt = "\n".join(f"[{n['id']}] {n['observation'][:60]}"
                            for n in remaining)
        obj = _gpt_json(model, SYSTEM_SUFFICIENCY,
                        f"Question: {qtext}\n\nChain:\n{chain_txt}\n\n"
                        f"Remaining:\n{rem_txt}\n\nIs the current chain "
                        f"sufficient?", max_tokens=200) or {}
        if obj.get("sufficient") or not obj.get("best_candidate_id"):
            break
        best_id = obj.get("best_candidate_id")
        nmap = {n["id"]: n for n in remaining}
        best = nmap.get(best_id, remaining[0])
        chain.append(best)
        remaining = [n for n in remaining if n["id"] != best["id"]]
    return chain


def _graph_to_text(chain, edges):
    """Render the evidence graph as a prompt-attachable text block."""
    if not chain:
        return ""
    idmap = {}
    lines = ["Structured visual evidence extracted from across the whole video "
             "(an evidence graph; use it to organize your reasoning):"]
    for n in sorted(chain, key=lambda x: x["t0"]):
        idmap[n["id"]] = _hms(n["t0"])
        lines.append(f"- [{_hms(n['t0'])}] {n['observation']}")
    rels = [e for e in (edges or []) if e["src"] in idmap and e["tgt"] in idmap]
    if rels:
        lines.append("Relations between evidence:")
        for e in rels:
            lines.append(f"- {idmap[e['src']]} --{e['relation']}--> "
                         f"{idmap[e['tgt']]}")
    return "\n".join(lines)


def _build_graph_cached(item, gparams, batch=10, max_side=448):
    """Build (or load from cache) the static clip-graph. Returns (chain, gdbg).
    On cache hit: NO scoring/edge GPT calls."""
    qid = str(item.get("_qid", "x")).replace("/", "_")
    cache_p = os.path.join(CACHE_DIR, f"{qid}.json")
    if os.path.exists(cache_p):
        try:
            j = json.load(open(cache_p))
            if j.get("params") == gparams:
                return j["chain"], j["graph_dbg"]
        except Exception:
            pass

    cap, fps, dur = _open(item["_abs_video"])
    if dur <= 0:
        cap.release()
        return [], {}
    qtext = _qtext(item)
    ci, sm = gparams["clip_interval"], gparams["score_model"]
    tks, mn = gparams["top_k_seeds"], gparams["max_nodes"]
    emc, mec = gparams["edge_min_confidence"], gparams["max_edge_candidates"]
    mc = gparams["max_chain"]

    # 1-2. cut + score
    clips = _make_clips(dur, ci)
    clip_scores = _score_clips(cap, fps, clips, qtext, batch, max_side, sm)
    clip_scores.sort(key=lambda x: x["score"], reverse=True)
    # 3. seeds
    seeds = [c for c in clip_scores if c["score"] >= 3][:tks]
    if not seeds:
        seeds = clip_scores[:tks]
    seed_t = {(c["t0"], c["t1"]) for c in seeds}
    candidates = [c for c in clip_scores
                  if (c["t0"], c["t1"]) not in seed_t and c["score"] >= 2][:mec]

    # 4. BFS edge discovery
    def mk(c, is_seed):
        return {"id": f"n{int(c['t0'])}", "t0": c["t0"], "t1": c["t1"],
                "observation": c["observation"], "score": c["score"],
                "frames": _clip_frames(cap, fps, c["t0"], c["t1"], 3, max_side),
                "is_seed": is_seed}

    nodes = {(c["t0"], c["t1"]): mk(c, True) for c in seeds}
    edges, edge_set, cand_nodes = [], set(), {}
    queue = deque(nodes.keys())
    while queue and len(nodes) < mn:
        ak = queue.popleft()
        A = nodes[ak]
        for c in candidates:
            ck = (c["t0"], c["t1"])
            if ck == ak:
                continue
            pk = tuple(sorted([ak, ck]))
            if pk in edge_set:
                continue
            edge_set.add(pk)
            if ck not in cand_nodes:
                cand_nodes[ck] = mk(c, False)
            e = _get_edge(cap, fps, A, cand_nodes[ck], qtext, max_side, sm)
            if e["relation"] != "unrelated" and e["confidence"] >= emc:
                if ck not in nodes and len(nodes) < mn:
                    nodes[ck] = cand_nodes[ck]
                    queue.append(ck)
                if ak in nodes and ck in nodes:
                    src, tgt = ((ck, ak) if e["direction"] == "B_to_A"
                                else (ak, ck))
                    edges.append({"src": nodes[src]["id"], "tgt": nodes[tgt]["id"],
                                  "relation": e["relation"],
                                  "confidence": e["confidence"]})

    # 5. minimal sufficient chain
    node_list = list(nodes.values())
    chain_nodes = _min_chain(node_list, qtext, mc, sm)
    if not chain_nodes:
        chain_nodes = sorted(node_list, key=lambda x: -x["score"])[:mc]
    chain = [{"id": n["id"], "t0": n["t0"], "t1": n["t1"],
              "observation": n["observation"], "score": n["score"]}
             for n in chain_nodes]
    gdbg = {"n_clips": len(clips), "seed_ts": [s["t0"] for s in seeds],
            "node_ts": sorted(n["t0"] for n in nodes.values()),
            "n_edges": len(edges), "edges": edges,
            "chain_ts": [n["t0"] for n in chain]}
    cap.release()
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump({"params": gparams, "chain": chain, "graph_dbg": gdbg},
              open(cache_p, "w"), ensure_ascii=False)
    return chain, gdbg


def _gparams(clip_interval, top_k_seeds, max_nodes, edge_min_confidence,
             max_edge_candidates, max_chain, score_model):
    return {"clip_interval": clip_interval, "top_k_seeds": top_k_seeds,
            "max_nodes": max_nodes, "edge_min_confidence": edge_min_confidence,
            "max_edge_candidates": max_edge_candidates,
            "max_chain": max_chain, "score_model": score_model}


def run_clip_graph(item, build_query, answer_fn, *,
                   clip_interval=30.0, top_k_seeds=3, max_nodes=8,
                   edge_min_confidence=3, max_edge_candidates=12,
                   max_chain=4, answer_frames=16, global_frames=8, batch=10,
                   score_model="gpt-5.4-mini", max_side=448):
    gparams = _gparams(clip_interval, top_k_seeds, max_nodes,
                       edge_min_confidence, max_edge_candidates, max_chain,
                       score_model)
    chain, gdbg = _build_graph_cached(item, gparams, batch, max_side)
    cap, fps, dur = _open(item["_abs_video"])
    if dur <= 0:
        cap.release()
        return "", {"error": "no_video"}

    # 6. answer with a FIXED budget = GLOBAL uniform frames (keep whole-video
    #    coverage) + chain-focused frames (evidence), merged in temporal order.
    chain = sorted(chain, key=lambda n: n["t0"])
    g = max(0, min(global_frames, answer_frames))
    chain_budget = answer_frames - g
    picked = []  # (ts, frame)
    if g > 0:                                   # global overview, spans 0..dur
        for ts in np.linspace(dur * 0.02, dur * 0.98, g).tolist():
            fr = _grab(cap, fps, ts, max_side)
            if fr is not None:
                picked.append((ts, fr))
    if chain and chain_budget > 0:              # focused chain-clip frames
        nfr = max(1, chain_budget // len(chain))
        for i, n in enumerate(chain):
            take = nfr + (1 if i < chain_budget - nfr * len(chain) else 0)
            if take <= 0:
                continue
            m = (n["t1"] - n["t0"]) * 0.1
            for ts in np.linspace(n["t0"] + m, n["t1"] - m, take).tolist():
                fr = _grab(cap, fps, ts, max_side)
                if fr is not None:
                    picked.append((ts, fr))
    cap.release()
    picked.sort(key=lambda x: x[0])
    picked = picked[:answer_frames]
    frames = [f for _, f in picked]
    ans_ts = [round(t, 1) for t, _ in picked]

    evidence = "\n".join(f"- {_hms(n['t0'])}: {n['observation']}"
                         for n in chain)
    query = (f"{build_query(item)}\n\nSelected visual evidence "
             f"(temporal order):\n{evidence}")
    answer = answer_fn(frames, query)

    return answer, {
        "clip_interval": clip_interval,
        "global_frames": g,
        **gdbg,
        "n_answer_frames": len(frames),
        "answer_ts": ans_ts,
    }


def _id_t0(nid):
    try:
        return float(str(nid)[1:])
    except (ValueError, TypeError):
        return None


def run_graph_focus(item, build_query, answer_fn, *, n_frames=64,
                    coverage_ratio=0.5, clip_interval=30.0, top_k_seeds=3,
                    max_nodes=8, edge_min_confidence=3, max_edge_candidates=12,
                    max_chain=4, score_model="gpt-5.4-mini", batch=10,
                    max_side=448):
    """Coverage base + graph-guided densification (THE clean test of edges):
    half the budget keeps full uniform coverage; the other half densifies the
    time windows of the CORE node + its EDGE-neighbors (the graph structure).
    No collapse, no second-hand text — only the graph decides WHERE to densify."""
    gparams = _gparams(clip_interval, top_k_seeds, max_nodes,
                       edge_min_confidence, max_edge_candidates, max_chain,
                       score_model)
    chain, gdbg = _build_graph_cached(item, gparams, batch, max_side)
    cap, fps, dur = _open(item["_abs_video"])
    if dur <= 0:
        cap.release()
        return "", {"error": "no_video"}
    cov = max(1, int(round(n_frames * coverage_ratio)))
    foc = n_frames - cov
    picked = []
    bounds = np.linspace(0, dur, cov + 1)              # coverage base
    for i in range(cov):
        picked.append((bounds[i] + bounds[i + 1]) / 2)

    foc_t0s = []                                        # core + edge-neighbors
    if chain:
        core_id = chain[0]["id"]
        foc_t0s = [chain[0]["t0"]]
        for e in gdbg.get("edges", []):
            if e["src"] == core_id:
                foc_t0s.append(_id_t0(e["tgt"]))
            elif e["tgt"] == core_id:
                foc_t0s.append(_id_t0(e["src"]))
        foc_t0s = sorted({t for t in foc_t0s if t is not None})
    if foc > 0 and foc_t0s:
        per = max(1, foc // len(foc_t0s))
        for t0 in foc_t0s:
            t1 = min(dur, t0 + clip_interval)
            m = (t1 - t0) * 0.1
            for ts in np.linspace(t0 + m, t1 - m, per).tolist():
                picked.append(ts)
    elif foc > 0:                                       # no graph -> more cover
        picked += np.linspace(0, dur, foc + 2).tolist()[1:-1]

    picked = sorted(picked)[:n_frames]
    frames = [fr for ts in picked
              if (fr := _grab(cap, fps, ts, max_side)) is not None]
    cap.release()
    if not frames:
        return "", {"error": "no_frames"}
    answer = answer_fn(frames, build_query(item))
    return answer, {"n_frames": len(frames), "cov": cov, "foc": foc,
                    "n_focus_nodes": len(foc_t0s),
                    "core_t0": chain[0]["t0"] if chain else None,
                    "n_edges": gdbg.get("n_edges"),
                    "node_ts": gdbg.get("node_ts")}


def _score_all_clips_cached(item, clip_interval, score_model, batch, max_side):
    """Score every clip (graph step 1-2 only, no edges). Returns (scores, dur).
    scores: list of {t0,t1,score,observation}. Cached per (qid, clip_interval)."""
    qid = str(item.get("_qid", "x")).replace("/", "_")
    # cache keyed by the scoring model; gpt-5.4-mini keeps the legacy name
    # (no suffix) so existing caches stay valid, others get their own.
    sfx = "" if score_model == "gpt-5.4-mini" else "_" + re.sub(r"[^A-Za-z0-9]", "", score_model)
    cache_p = os.path.join(CACHE_DIR, f"scores_{qid}_ci{int(clip_interval)}{sfx}.json")
    if os.path.exists(cache_p):
        try:
            j = json.load(open(cache_p))
            return j["scores"], j["dur"]
        except Exception:
            pass
    cap, fps, dur = _open(item["_abs_video"])
    if dur <= 0:
        cap.release()
        return [], 0
    qtext = _qtext(item)
    clips = _make_clips(dur, clip_interval)
    cs = _score_clips(cap, fps, clips, qtext, batch, max_side, score_model)
    cap.release()
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump({"scores": cs, "dur": dur}, open(cache_p, "w"),
              ensure_ascii=False)
    return cs, dur


def run_graph_select(item, build_query, answer_fn, *, n_frames=16,
                     clip_interval=20.0, score_model="gpt-5.4-mini",
                     batch=10, max_side=448):
    """Coverage-preserving, score-guided frame selection: split the video into
    n_frames equal segments (same coverage as uniform), but inside each segment
    pick the frame from the HIGHEST-scoring clip (graph's relevance score)
    instead of the segment midpoint. No collapse, no second-hand text."""
    cs, dur = _score_all_clips_cached(item, clip_interval, score_model, batch,
                                      max_side)
    cap, fps, dur2 = _open(item["_abs_video"])
    if dur2 <= 0:
        cap.release()
        return "", {"error": "no_video"}
    dur = dur or dur2
    bounds = np.linspace(0, dur, n_frames + 1)
    chosen, n_optimized = [], 0
    for i in range(n_frames):
        lo, hi = bounds[i], bounds[i + 1]
        inb = [c for c in cs if lo <= c["t0"] < hi]
        if inb:
            best = max(inb, key=lambda c: c.get("score", 1))
            if best.get("score", 1) >= 3:          # only trust real signal
                chosen.append((best["t0"] + best["t1"]) / 2)
                n_optimized += 1
                continue
        chosen.append((lo + hi) / 2)               # fallback: uniform midpoint
    frames = []
    for ts in chosen:
        fr = _grab(cap, fps, ts, max_side)
        if fr is not None:
            frames.append(fr)
    cap.release()
    if not frames:
        return "", {"error": "no_frames"}
    answer = answer_fn(frames, build_query(item))
    return answer, {"n_frames": len(frames), "n_optimized": n_optimized,
                    "clip_interval": clip_interval,
                    "chosen_ts": [round(t) for t in chosen]}


def run_graph_snapshot(item, build_query, answer_fn, *, n_frames=16,
                       snap_k=3, snap_n=3, snap_sec=2.0, clip_interval=20.0,
                       score_model="gpt-5.4-mini", batch=10, max_side=448):
    """Coverage base + SNAPSHOT densification. Keep uniform
    coverage for most of the budget, plus the top-snap_k highest-scoring moments
    each rendered as a DENSE ±snap_sec mini-clip (snap_n frames) so the model
    can actually see the action at those key moments — not a single blurry frame."""
    cs, dur = _score_all_clips_cached(item, clip_interval, score_model, batch,
                                      max_side)
    cap, fps, dur2 = _open(item["_abs_video"])
    if dur2 <= 0:
        cap.release()
        return "", {"error": "no_video"}
    dur = dur or dur2
    # top-snap_k high-score moments, spread out (avoid overlapping snapshots)
    keys = []
    for c in sorted(cs, key=lambda x: -x.get("score", 1)):
        if c.get("score", 1) < 3:
            break
        ct = (c["t0"] + c["t1"]) / 2
        if all(abs(ct - k) > snap_sec * 2 for k in keys):
            keys.append(ct)
        if len(keys) >= snap_k:
            break
    cov = max(0, n_frames - len(keys) * snap_n)
    # coverage base = graph_select-style: split into cov segments, each picks
    # its highest-scoring frame (so the base IS graph_select, not plain uniform)
    picked = []
    if cov > 0:
        bnd = np.linspace(0, dur, cov + 1)
        for i in range(cov):
            lo, hi = bnd[i], bnd[i + 1]
            inb = [c for c in cs if lo <= c["t0"] < hi]
            best = max(inb, key=lambda c: c.get("score", 1)) if inb else None
            if best and best.get("score", 1) >= 3:
                picked.append((best["t0"] + best["t1"]) / 2)
            else:
                picked.append((lo + hi) / 2)
    for ct in keys:
        lo, hi = max(0, ct - snap_sec), min(dur, ct + snap_sec)
        picked += list(np.linspace(lo, hi, snap_n))
    picked = sorted(float(t) for t in picked)[:n_frames]
    frames = [fr for ts in picked
              if (fr := _grab(cap, fps, ts, max_side)) is not None]
    cap.release()
    if not frames:
        return "", {"error": "no_frames"}
    answer = answer_fn(frames, build_query(item))
    return answer, {"n_frames": len(frames), "n_snap": len(keys), "cov": cov,
                    "snap_n": snap_n, "snap_sec": snap_sec,
                    "snap_ts": [round(k) for k in keys]}


def run_uniform_gtext(item, build_query, answer_fn, *, uniform_frames=64,
                      clip_interval=30.0, top_k_seeds=3, max_nodes=8,
                      edge_min_confidence=3, max_edge_candidates=12,
                      max_chain=4, batch=10, score_model="gpt-5.4-mini",
                      max_side=448):
    """Coverage-FIRST + graph-as-organizer: answer over uniform_frames frames
    (full coverage, unchanged) PLUS the evidence graph rendered as TEXT — the
    graph organizes evidence without spending any of the frame budget."""
    from frames import sample_uniform
    gparams = _gparams(clip_interval, top_k_seeds, max_nodes,
                       edge_min_confidence, max_edge_candidates, max_chain,
                       score_model)
    chain, gdbg = _build_graph_cached(item, gparams, batch, max_side)
    frames = sample_uniform(item["_abs_video"], n=uniform_frames)
    if not frames:
        return "", {"error": "no_frames"}
    gtext = _graph_to_text(chain, gdbg.get("edges", []))
    query = build_query(item)
    if gtext:
        query = f"{query}\n\n{gtext}"
    answer = answer_fn(frames, query)
    return answer, {"uniform_frames": len(frames), "gtext_used": bool(gtext),
                    "n_chain": len(chain), **gdbg}
