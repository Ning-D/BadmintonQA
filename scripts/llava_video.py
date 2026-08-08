"""LLaVA-Video-7B-Qwen2 answering backbone -- architecturally heterogeneous 3rd backbone
(SigLIP vision encoder + Qwen2-7B LLM + LLaVA-style MLP projector, frames fed as the
'video' modality). Exposes the SAME answer_mc(frames, query, max_new_tokens, labels)
interface as qwen_vl.QwenVL / mplug_owl3.MplugOwl3 / internvl.InternVL, so it drops into
run_vmme / run_evicover_joint / run_lvnet. MUST run in the `llava_video` conda env
(LLaVA-NeXT `llava` package + transformers 4.37). Handles 32/64 frames natively
(unlike InternVL which overflows context at 32).
"""
import os, sys, copy, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
# llava is a local (non-pip-installed) package living in the LLaVA-NeXT repo
_LLAVA_NEXT = os.environ.get("LLAVA_NEXT_DIR", "/mnt/HDD12TB-1/ding_2026/LLaVA-NeXT")
if _LLAVA_NEXT not in sys.path:
    sys.path.insert(0, _LLAVA_NEXT)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

MODEL_PATH = os.environ.get("LLAVAVIDEO_MODEL", "lmms-lab/LLaVA-Video-7B-Qwen2")


class LLaVAVideo:
    def __init__(self, model_path=MODEL_PATH, device="cuda"):
        self.tok, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", torch_dtype="bfloat16", device_map="auto",
            attn_implementation="sdpa")
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def answer_mc(self, frames, query, max_new_tokens=64, labels=None):
        """frames: list of RGB HxWx3 arrays (already sampled by the selector)."""
        arr = np.stack([np.asarray(f) for f in frames])                 # [N,H,W,3] RGB
        video = self.image_processor.preprocess(arr, return_tensors="pt")["pixel_values"]
        video = [video.to(self.device).bfloat16()]
        q = DEFAULT_IMAGE_TOKEN + "\n" + query
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tok, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(self.device)
        out = self.model.generate(input_ids, images=video, modalities=["video"],
                                  do_sample=False, temperature=0, max_new_tokens=max_new_tokens)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
