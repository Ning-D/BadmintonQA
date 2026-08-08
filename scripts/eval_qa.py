#!/usr/bin/env python3
"""Compute BadmintonQA accuracy from a predictions file.

Usage:
    python scripts/eval_qa.py predictions.json

predictions.json maps question id -> predicted option letter:
    {"<question id>": "A", "<question id>": "C", ...}

Prints overall accuracy and a per-capability breakdown
(Recognition / Counting / Temporal Reasoning / Grounding / Tactical).
Unanswered questions count as wrong.
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VQA_DIR = os.path.join(ROOT, "badminton_data", "vqa")


def load_qas():
    """Flatten both QA files into [{id, answer, capability}] (533 questions)."""
    qas = []
    for q in json.load(open(os.path.join(VQA_DIR, "taxonomy_qa_v1.json"))):
        qas.append({"id": q["id"], "answer": q["answer"], "capability": q["capability"]})
    for match in json.load(open(os.path.join(VQA_DIR, "tactic_qa_all.json"))):
        for q in match["qa"]:
            qas.append({"id": q["id"], "answer": q["answer"], "capability": "Tactical"})
    return qas


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip())
    preds = json.load(open(sys.argv[1]))

    qas = load_qas()
    stats = defaultdict(lambda: [0, 0])  # capability -> [n_correct, n_total]
    answered = 0
    for q in qas:
        pred = str(preds.get(q["id"], "")).strip().upper()[:1]
        answered += q["id"] in preds
        correct = int(pred == q["answer"])
        for key in (q["capability"], "Overall"):
            stats[key][0] += correct
            stats[key][1] += 1

    order = ["Recognition", "Counting", "Temporal Reasoning", "Grounding", "Tactical", "Overall"]
    print(f"answered {answered}/{len(qas)} questions\n")
    for cap in order:
        n_correct, n_total = stats[cap]
        print(f"{cap:<20} {100.0 * n_correct / n_total:5.1f}  ({n_correct}/{n_total})")


if __name__ == "__main__":
    main()
