# BadmintonQA: A Long-Video Question Answering Dataset on Full Badminton Matches

> 📢 **News**: This paper has been accepted by **MMSports 2026** (ACM International Workshop on Multimedia Content Analysis in Sports).

**BadmintonQA** is a long-video question-answering benchmark built on **full-length badminton broadcast videos** (1–2 hours per match): **533 four-choice questions** over **19 full matches** (12 men's singles + 7 men's doubles) from four BWF World Tour tournaments. Questions target match-level understanding — score progression, shot-type statistics, and **cross-game tactical evolution** — so answering them requires locating evidence scattered across an entire broadcast rather than a single clip.

## QA data

The 533 questions live in two JSON files under `badminton_data/vqa/`:

| File | Questions | Content |
|---|---|---|
| `taxonomy_qa_v1.json` | 446 | Capability-taxonomy questions: **Recognition** (162), **Counting** (165), **Temporal Reasoning** (79), **Grounding** (40) |
| `tactic_qa_all.json` | 87 | **Tactical**-evolution questions (cross-game style shift, tactic frequency/rose, adaptation), grouped per match |

Each question is four-choice with a single correct option and comes with evidence pointers (game / rally / frame) and an explanation:

```jsonc
{
  "id": "KAPAL-API-Indonesia-Open-2025-...-F::one_shot::1",
  "match_name": "KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F",
  "capability": "Recognition",
  "question": "In game 1 of the full broadcast match, around broadcast time 09:01, Chou Tien Chen plays a shot. What type of shot is it?",
  "options": ["smash", "lift", "net shot", "clear"],
  "answer": "A",
  "evidence": {"game": 1, "rally": 0, "frame": 16253, "single_moment": true},
  "explanation": "hit_inferred labels the contact at frame 16253 as 'smash'."
}
```

In `tactic_qa_all.json` the questions are nested under one object per match (`{"match_name": ..., "qa": [...]}`), with the same question fields.

**Videos** are not redistributed — the 19 source broadcasts are publicly available on YouTube ([BWF TV](https://www.youtube.com/c/bwftv)), identified by the `match_name` field of each question.

## Computing QA accuracy

Run your model over the questions and write its answers to a JSON file mapping question `id` → option letter:

```json
{"KAPAL-API-Indonesia-Open-2025-...-F::one_shot::1": "A", "...": "C"}
```

Then score it (no dependencies beyond the standard library; unanswered questions count as wrong):

```bash
python scripts/eval_qa.py predictions.json
```

```
answered 533/533 questions

Recognition           ...
Counting              ...
Temporal Reasoning    ...
Grounding             ...
Tactical              ...
Overall               ...
```

## License and citation

Annotations and code are released for research use. Video copyright belongs to [BWF TV](https://www.youtube.com/c/bwftv); do not redistribute the videos.

The citation entry will be added once the MMSports 2026 proceedings are published.
