# BadmintonQA: A Long-Video Question Answering Dataset on Full Badminton Matches

> 📢 **News**: This paper has been accepted by **MMSports 2026** (ACM International Workshop on Multimedia Content Analysis in Sports).

**BadmintonQA** is a long-video question-answering benchmark built on **full-length badminton broadcast videos** (1–2 hours per match): **533 four-choice questions** over **19 full matches** (12 men's singles + 7 men's doubles) from four BWF World Tour tournaments. Questions target match-level understanding — score progression, shot-type statistics, and **cross-game tactical evolution** — so answering them requires locating evidence scattered across an entire broadcast rather than a single clip.

The benchmark comes with a two-phase evaluation protocol (keyframe selection → frozen VLM answering) and **EviCover**, our training-free, backbone-agnostic frame selector that first divides a video into temporal segments, then selects the segments most useful for distinguishing the answer options based on their captions, and finally retrieves one question-relevant frame from each selected segment.

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

## Getting the videos

Videos are not redistributed (copyright: [BWF TV](https://www.youtube.com/c/bwftv)). Download the 19 source broadcasts from YouTube into the paths the scripts expect (`badminton_data/videos/<tournament>/` and `badminton_data/videos_doubles/`):

```bash
pip install yt-dlp        # ffmpeg required for stream merging
cd badminton_data
python download_youtube.py --list           # show the plan
python download_youtube.py                  # download all 19 matches (~15 GB, ≤720p mp4)
python download_youtube.py --match Antonsen # only matches containing a substring
```

`Badminton_video_list.csv` holds the YouTube link for every match, each verified against the BWF TV channel with an exact duration match. All annotations index frames in 0-based decode order on the 1280×720 / 30 fps stream, so keep the ≤720p mp4 format the script enforces.

## Computing QA accuracy

Run your model over the questions and write its answers to a JSON file mapping question `id` → option letter:

```json
{"KAPAL-API-Indonesia-Open-2025-...-F::one_shot::1": "A", "...": "C"}
```

Then score it (standard library only; unanswered questions count as wrong):

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

## Running the methods

All methods follow the same **two-phase protocol**: phase 1 selects N ∈ {8, 16, 32, 64} keyframes per question and writes a cache (`badminton_data/vqa/table2/*_cache_f{N}.json`); phase 2 lets a frozen VLM answer from exactly those frames. Every step is cached and resumable — rerunning skips finished questions — and `--shard i/k` splits any step across GPUs/processes by match.

### Environment

The answering backbones follow their upstream requirements and run in separate conda environments:

| Env | Used for | Key packages |
|---|---|---|
| `qwen25vl` | phase-1 selectors + Qwen3-VL-8B-Instruct + mPLUG-Owl3-7B | `torch`, `transformers`, `opencv-python`, `numpy`, `scikit-learn`, `qwen-vl-utils` |
| `llava_video` | LLaVA-Video-7B-Qwen2 | per [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) upstream; set `LLAVA_NEXT_DIR` to your LLaVA-NeXT clone |

Models are pulled from Hugging Face on first use and can be overridden via environment variables: `QWEN_MODEL` (default `Qwen/Qwen3-VL-8B-Instruct`), `MPLUG_MODEL` (`mPLUG/mPLUG-Owl3-7B-241101`), `LLAVAVIDEO_MODEL` (`lmms-lab/LLaVA-Video-7B-Qwen2`), `BLIP_MODEL` (`Salesforce/blip-itm-large-coco`). A single GPU with ≥24 GB memory is enough for every step (7–8B VLMs at fp16).

The LLM-assisted selectors (EviCover captioning/selection, LVNet keyword extraction) call the OpenAI API (`gpt-5.4-mini`):

```bash
export OPENAI_API_KEY=sk-...
```

Below, `$QWEN_PY` / `$LLAVA_PY` denote the python of the corresponding conda env.

### Uniform (question-agnostic reference) — selection + answering in one step

```bash
CUDA_VISIBLE_DEVICES=0 $QWEN_PY scripts/run_table2.py --backbone qwen3vl --frames 32
```

### AKS (question-aware baseline)

```bash
$QWEN_PY scripts/run_aks_full.py --phase 1 --frames 8,16,32,64      # BLIP scoring + AKS selection
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector aks --frames 32 --backbone qwen3vl
```

### VideoTree / BOLT / LVNet baselines

```bash
$QWEN_PY scripts/run_baselines_full.py                              # phase 1: all three, all budgets
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector videotree --frames 32 --backbone qwen3vl
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector bolt      --frames 32 --backbone qwen3vl
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector lvnet     --frames 32 --backbone qwen3vl
```

### EviCover (ours)

```bash
$QWEN_PY scripts/run_evicover.py --budgets 8,16,32,64               # phase 1: event nodes + LLM selection
                                                                    # (also writes the EviCover-U caches)
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector evicover  --frames 32 --backbone qwen3vl
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector evicoveru --frames 32 --backbone qwen3vl  # ablation: uniform over event nodes
```

For the `llava` backbone replace the phase-2 python with `$LLAVA_PY`; for `mplug` keep `$QWEN_PY`. Per-run accuracy JSONs land in `badminton_data/vqa/table2/{selector}_f{N}_{backbone}.json`.

## Baseline implementations

The baseline selectors are re-implementations adapted from their official repositories — please refer to (and cite) the original works:

| Baseline | Adapted from | Paper |
|---|---|---|
| AKS | [ncTimTang/AKS](https://github.com/ncTimTang/AKS) | Adaptive Keyframe Sampling for Long Video Understanding (CVPR 2025) |
| VideoTree | [Ziyang412/VideoTree](https://github.com/Ziyang412/VideoTree) | VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos (CVPR 2025) |
| BOLT | [sming256/BOLT](https://github.com/sming256/BOLT) | BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding (CVPR 2025) |
| LVNet | [jongwoopark7978/LVNet](https://github.com/jongwoopark7978/LVNet) | Too Many Frames, Not All Useful: Efficient Strategies for Long-Form Video QA (EACL 2026) |

`scripts/aks_select.py` keeps AKS's `meanstd` selection verbatim; `scripts/videotree_select.py` keeps VideoTree's breadth/depth clustering structure over lighter features (CLIP-L + BLIP-ITM relevance); BOLT is inverse transform sampling over the BLIP relevance distribution; LVNet ranks frames by BLIP alignment to LLM-extracted option keywords.

## License and citation

Annotations and code are released for research use. Video copyright belongs to [BWF TV](https://www.youtube.com/c/bwftv); do not redistribute the videos.

The citation entry will be added once the MMSports 2026 proceedings are published.
