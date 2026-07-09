# BadmintonQA (BTE-VQA)

整场羽毛球长视频 VQA 数据集与流水线（战术演变跨 game 题为主）。
2026-07 从 `SportsCOT` 仓库拆出（原历史见 `/mnt/HDD12TB-1/ding/SportsCOT` 的 git log）。

## 目录

- `badminton_data/` — 数据集本体：`videos/`（符号链接）、`annotations/`（含 `gen_tactic_qa.py` → `vqa/`）、`vqa/`、`viz/`（rally 剪辑，不入 git）
- `data_pre/` — 预处理与 QA 生成：rally captions、evomem taxonomy、`shot-example/`（detective / qceg / archive 三代 VQA 生成）、球/人轨迹插值
- `scripts/` — 从 `SportsCOT/videoespresso/` 迁来的羽毛球脚本：
  - `badminton_data.py` — 数据加载 + build_query
  - `run_tactic_qa_aks.py` — tactic QA 的 AKS 选帧评测
  - `prewarm_badminton_scores.py` — CLIP 分数预热（依赖同目录 `graph_clip.py`，为 EviGraph `videoespresso/graph_clip.py` 的拷贝）
  - `court_*.py` / `make_court_editor.py` — 球场标定
- `models/`, `datasets/`, `train_ddp.py`, `main_ddp.py`, `caption_metrics.py` — 早期 caption 训练遗产（VideoMAE / MatchVision baseline）

## 与 EviGraph 的关系

EviGraph 主仓库 `SportsCOT/videoespresso/run_vmme.py` 仍保留 `--dataset badminton` 分支（懒加载 `badminton_data`）；在那边跑需要 `PYTHONPATH` 加上本仓库 `scripts/`。

## 注意

- hit 的 near/far 用得分推断（发球 = 上回合胜方）；hit 轨迹约 30% 缺帧。
