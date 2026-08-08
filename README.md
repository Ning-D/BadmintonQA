# BadmintonQA: A Long-Video Question Answering Dataset on Full Badminton Matches

> 📢 **News**: This paper has been accepted by **MMSports 2026** (ACM International Workshop on Multimedia Content Analysis in Sports).

Understanding a full sports match requires reasoning about how patterns of play, match dynamics, and competitors' behaviors evolve over time. **BadmintonQA** is a long-video question-answering benchmark built on **full-length badminton broadcast videos** (1–2 hours per match): **533 four-choice questions** over **19 full matches** (12 men's singles + 7 men's doubles) from four BWF World Tour tournaments. Questions target match-level understanding — score progression, shot-type statistics, and **cross-game tactical evolution** — so answering them requires locating evidence scattered across an entire broadcast rather than a single clip.

The benchmark ships with a two-phase evaluation protocol (keyframe selection → frozen VLM answering) and **EviCover**, our training-free, backbone-agnostic frame selector that first divides a video into temporal segments, then selects the segments most useful for distinguishing the answer options based on their captions, and finally retrieves one question-relevant frame from each selected segment.

## Repository layout

```
BadmintonQA/
├── BFMD_data/           # dataset package: annotations + video download & visualization tools
│   ├── README.md        #   full annotation format documentation
│   ├── download_youtube.py
│   └── annotations/     #   metadata / court / pose / shuttle / shot_type / hit_inferred ...
├── badminton_data/      # working data root used by the experiment scripts
│   └── vqa/             #   QA sets + per-selector caches + results (table2/)
├── scripts/             # experiment code (frame selectors + answering)
├── viz/                 # paper figure scripts
└── docs/                # method design and result notes
```

## Dataset

The annotation package under [`BFMD_data/`](BFMD_data/) covers **1,058 rallies and 11,301 shots** across the 19 matches, including match metadata (games / rallies / score events), court geometry, player boxes and 17-keypoint poses, shuttlecock trajectories, per-shot type labels, inferred hit events, and per-hit natural-language captions. See [`BFMD_data/README.md`](BFMD_data/README.md) for formats and coordinate conventions, and `BFMD_data/vis.py` for rendering all annotations onto the videos.

The QA sets live at `badminton_data/vqa/taxonomy_qa_v1.json` (capability-taxonomy questions) and `badminton_data/vqa/tactic_qa_all.json` (tactical-evolution questions) — 533 four-option multiple-choice questions in total.

## Environment setup

The answering backbones follow their upstream requirements and run in **separate conda environments**:

| Env | Used for | Key packages |
|---|---|---|
| `qwen25vl` | phase-1 selectors + Qwen3-VL-8B-Instruct + mPLUG-Owl3-7B | `torch`, `transformers`, `opencv-python`, `numpy`, `qwen-vl-utils` |
| `llava_video` | LLaVA-Video-7B-Qwen2 | per LLaVA-Video upstream |

Phase-1 selection additionally uses BLIP-ITM (`Salesforce/blip-itm-large-coco`), SigLIP (`google/siglip-so400m-patch14-384`) and CLIP-L — all pulled automatically from Hugging Face on first run. A GPU with ≥24 GB memory is enough for every step (7–8B VLMs at fp16).

LLM-assisted selectors (EviCover captioning/selection, LVNet keyword extraction) call the OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
```

## Data preparation

1. **Annotations** ship with the repository under `BFMD_data/annotations/` (see `BFMD_data/README.md`).
2. **Videos** are not redistributed (copyright: [BWF TV](https://www.youtube.com/c/bwftv)). Download them from YouTube into the expected paths:

```bash
pip install yt-dlp        # ffmpeg required
cd BFMD_data
python download_youtube.py --list    # show the plan
python download_youtube.py           # download all 19 matches (~15 GB, ≤720p mp4)
```

3. **QA loading** is handled by `scripts/run_table2.py:load_qas`, which drops questions whose video is missing.

## Running the main experiments

All runs follow the same **two-phase protocol**. Phase 1 selects N ∈ {8, 16, 32, 64} keyframes per question and writes a cache (`badminton_data/vqa/table2/*_cache_f{N}.json`); phase 2 lets a frozen VLM answer from exactly those frames. Every step is cached and resumable — rerunning skips finished questions. `--shard i/k` splits any step across GPUs/processes by match.

Let `QWEN_PY` / `LLAVA_PY` denote the python of the corresponding conda env.

**Uniform selector (question-agnostic reference), selection + answering in one step:**

```bash
CUDA_VISIBLE_DEVICES=0 $QWEN_PY scripts/run_table2.py --backbone qwen3vl --frames 32
```

**AKS (question-aware baseline):**

```bash
$QWEN_PY scripts/run_aks_full.py --phase 1                       # BLIP scoring + AKS selection
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector aks --frames 32 --backbone qwen3vl
```

**VideoTree / BOLT / LVNet baselines (phase 1), then answer via run_aks_full.py:**

```bash
$QWEN_PY scripts/run_baselines_full.py                           # all three, all budgets
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector videotree --frames 32 --backbone qwen3vl
```

**EviCover (ours):**

```bash
$QWEN_PY scripts/run_evicover.py --budgets 8,16,32,64            # event nodes + LLM selection
$QWEN_PY scripts/run_aks_full.py --phase 2 --selector evicover --frames 32 --backbone qwen3vl
```

For the `llava` backbone replace the phase-2 python with `$LLAVA_PY`. Batch drivers for full sweeps: `scripts/run_baselines_queue.sh`, `scripts/run_ablation_queue.sh`.

## Collecting results

Per-run accuracy JSONs land in `badminton_data/vqa/table2/{selector}_f{N}_{backbone}.json`. Aggregate them into the paper tables:

```bash
python scripts/fill_results.py      # overall / per-capability accuracy
python scripts/fill_table2.py       # Table 2 (selector x budget x backbone)
```

## License and citation

Annotations and code are released for research use. Video copyright belongs to [BWF TV](https://www.youtube.com/c/bwftv); do not redistribute the videos.

```bibtex
@inproceedings{ding2026badmintonqa,
  title     = {BadmintonQA: A Long-Video Question Answering Dataset on Full Badminton Matches},
  author    = {Ding, Ning and Fujii, Keisuke},
  booktitle = {Proceedings of the ACM International Workshop on Multimedia Content Analysis in Sports (MMSports)},
  year      = {2026}
}
```
