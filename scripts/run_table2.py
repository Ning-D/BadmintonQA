"""
Table 2 runner: Overall accuracy on the full BadmintonQA set (taxonomy_qa_v1 +
tactic_qa_all) with the UNIFORM keyframe selector across answering backbones.

Uniform selection is question-agnostic: N frames evenly spaced over the whole
broadcast, decoded ONCE per match and reused for every question of that match.

Backbones (paper Table 2): qwen3vl = Qwen3-VL-8B-Instruct, mplug = mPLUG-Owl3-
7B-241101, llava = LLaVA-Video-7B-Qwen2. Wrappers live in this directory
(qwen_vl.py / mplug_owl3.py / llava_video.py); each backbone runs in its own
conda env (see README).

Usage:
  CUDA_VISIBLE_DEVICES=0 <env-python> run_table2.py --backbone qwen3vl
  ... --frames 32 --shard 0/2 (optional sharding by match)
Results: badminton_data/vqa/table2/uniform_f{N}_{backbone}[.shard].json
Resume: existing rows are kept; finished QAs are skipped.
"""
import os, re, sys, json, glob, time, argparse
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "badminton_data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LET = "ABCD"


def _fix_video(v):
    """Rebase the absolute video path stored in the QA json onto this repo."""
    if v and not os.path.exists(v) and "badminton_data/" in v:
        cand = os.path.join(ROOT, v.split("badminton_data/", 1)[1])
        if os.path.exists(cand):
            return cand
    return v


def load_qas():
    items = json.load(open(f"{ROOT}/vqa/taxonomy_qa_v1.json"))
    for q in items:
        q["video"] = _fix_video(q.get("video", ""))
    for m in json.load(open(f"{ROOT}/vqa/tactic_qa_all.json")):
        for q in m["qa"]:
            items.append({"id": q["id"], "match_name": m["match_name"],
                          "video": _fix_video(m.get("video", "")),
                          "capability": "Tactical Reasoning", "type": q["type"],
                          "question": q["question"], "options": q["options"],
                          "answer": q["answer"]})
    out = [x for x in items if x.get("video") and os.path.exists(x["video"])]
    if len(out) != len(items):
        print(f"[table2] WARNING: {len(items)-len(out)} QAs dropped (no video)")
    return out


def prompt_of(q):
    opts = "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(q["options"]))
    return ("These frames are uniformly sampled across a full badminton match "
            "broadcast (multiple games, in chronological order). Answer the "
            "multiple-choice question about this match.\n"
            f"Question: {q['question']}\nOptions:\n{opts}\n"
            "Answer with the single letter (A, B, C, or D) only.")


def parse_letter(raw):
    m = re.search(r"[ABCD]", (raw or "").upper())
    return m.group(0) if m else None


def uniform_frames(video, n, max_side=512):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total - 2), n).astype(int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        h, w = bgr.shape[:2]
        s = max_side / max(h, w)
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def load_backbone(name, device):
    if name == "qwen3vl":
        os.environ.setdefault("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        from qwen_vl import QwenVL
        return QwenVL(device=device)
    if name == "mplug":
        from mplug_owl3 import MplugOwl3
        return MplugOwl3(device=device)
    if name == "llava":
        from llava_video import LLaVAVideo
        return LLaVAVideo(device=device)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["qwen3vl", "mplug", "llava"])
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/k shard by match")
    args = ap.parse_args()
    si, sk = map(int, args.shard.split("/"))

    qas = load_qas()
    by_video = defaultdict(list)
    for q in qas:
        by_video[q["video"]].append(q)
    videos = sorted(by_video)
    videos = [v for i, v in enumerate(videos) if i % sk == si]

    os.makedirs(f"{ROOT}/vqa/table2", exist_ok=True)
    tag = f"uniform_f{args.frames}_{args.backbone}" + (f".s{si}of{sk}" if sk > 1 else "")
    outp = f"{ROOT}/vqa/table2/{tag}.json"
    rows = json.load(open(outp)) if os.path.exists(outp) else []
    done = {r["id"] for r in rows}
    todo_n = sum(1 for v in videos for q in by_video[v] if q["id"] not in done)
    print(f"[table2] {args.backbone} f={args.frames} shard {args.shard}: "
          f"{len(videos)} matches, {todo_n} QAs to run ({len(done)} cached)", flush=True)
    if todo_n == 0:
        report(rows, outp)
        return

    vlm = load_backbone(args.backbone, args.device)
    for vi, v in enumerate(videos):
        qs = [q for q in by_video[v] if q["id"] not in done]
        if not qs:
            continue
        t0 = time.time()
        frames = uniform_frames(v, args.frames)
        if not frames:
            print(f"[table2] !! no frames {v}")
            continue
        for q in qs:
            try:
                raw = vlm.answer_mc(frames, prompt_of(q))
            except Exception as e:
                raw = f"ERROR:{e}"
            pred = parse_letter(raw)
            rows.append({"id": q["id"], "match": q["match_name"],
                         "capability": q["capability"], "type": q["type"],
                         "gold": q["answer"], "pred": pred,
                         "correct": pred == q["answer"],
                         "raw": str(raw)[:80]})
        json.dump(rows, open(outp, "w"))
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"[table2] [{vi+1}/{len(videos)}] {os.path.basename(v)[:44]:44} "
              f"{len(qs)} QA in {time.time()-t0:.0f}s | running acc {acc:.3f}",
              flush=True)
    report(rows, outp)


def report(rows, outp):
    n = len(rows)
    nc = sum(r["correct"] for r in rows)
    print(f"\n=== {os.path.basename(outp)}: {nc}/{n} = {nc/max(1,n)*100:.1f}% "
          f"(random 25.0) ===")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["capability"]][0] += r["correct"]
        by[r["capability"]][1] += 1
    for c, (a, b) in sorted(by.items()):
        print(f"  {c:22} {a:3}/{b:<3} = {a/b*100:.1f}%")
    json.dump(rows, open(outp, "w"))


if __name__ == "__main__":
    main()
