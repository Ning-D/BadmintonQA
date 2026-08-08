"""
Stage A of the AKS baseline: per-frame BLIP-ITM relevance scoring.

Faithful to ncTimTang/AKS feature_extract.py (https://github.com/ncTimTang/AKS):
sample 1 fps, score every frame against the question with BLIP image-text-
matching (softmax match prob). We use HuggingFace `Salesforce/blip-itm-large-coco`
(== AKS's BLIP-large-COCO weights) to avoid the LAVIS install. The ViT image
encoding is done once per video and reused across that video's questions; only
the (cheap) text cross-attention runs per question.
"""
import os
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import BlipForImageTextRetrieval, AutoProcessor

_LOCAL = "/mnt/HDD12TB-1/ding_2026/models/blip-itm-large-coco-st"
MODEL_ID = os.environ.get(
    "BLIP_MODEL", _LOCAL if os.path.isdir(_LOCAL) else "Salesforce/blip-itm-large-coco")


def read_1fps(video_path, max_side=512, max_pool=0):
    """Sequentially decode and keep ~1 frame/sec. Returns (pil_frames, frame_idx)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(fps)))
    frames, idx, i = [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if i % step == 0:
            h, w = bgr.shape[:2]
            s = max_side / max(h, w)
            if s < 1.0:
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            idx.append(i)
        i += 1
    cap.release()
    if max_pool and len(frames) > max_pool:          # subsample very long videos
        keep = np.linspace(0, len(frames) - 1, max_pool).round().astype(int)
        frames = [frames[k] for k in keep]
        idx = [idx[k] for k in keep]
    return frames, idx


@torch.no_grad()
def encode_vision(model, proc, frames, device, bs=48):
    embs = []
    for k in range(0, len(frames), bs):
        pv = proc(images=frames[k:k + bs], return_tensors="pt")["pixel_values"].to(device, torch.float16)
        embs.append(model.vision_model(pv)[0].cpu())     # [b, seq, dim] -> CPU
    return torch.cat(embs, 0)                            # [F, seq, dim] fp16 on CPU


@torch.no_grad()
def score_question(model, proc, image_embeds, question, device, bs=64):
    tok = proc(text=question, return_tensors="pt", truncation=True, max_length=35).to(device)
    F = image_embeds.shape[0]
    out = np.empty(F, dtype=np.float32)
    for k in range(0, F, bs):
        ie = image_embeds[k:k + bs].to(device, torch.float16)
        iatt = torch.ones(ie.shape[:-1], dtype=torch.long, device=device)
        ids = tok.input_ids.expand(ie.shape[0], -1)
        am = tok.attention_mask.expand(ie.shape[0], -1)
        te = model.text_encoder(input_ids=ids, attention_mask=am,
                                encoder_hidden_states=ie, encoder_attention_mask=iatt,
                                return_dict=True)
        logits = model.itm_head(te.last_hidden_state[:, 0, :])    # [b,2]
        out[k:k + ie.shape[0]] = logits.float().softmax(-1)[:, 1].cpu().numpy()
    return out.tolist()
