"""
mPLUG-Owl3-7B-241101 wrapper — an alternative answering backbone, used to run
the open-source 7B SOTA peer (LVBench leaderboard: mPLUG-Owl3-7B @64 = 43.5)
OURSELVES, on the SAME videos/frames/protocol as our method, so the comparison
is apples-to-apples (no coverage / subtitle / version / harness mismatch).

Only answer_mc is needed (uniform + snapshot answering); frame selection for
snapshot is done by gpt-5.4-mini upstream, backbone-agnostic. Mirrors the
QwenVL.answer_mc(frames, query) interface: frames = list of HxWx3 RGB arrays.
"""
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

_LOCAL = "/mnt/HDD12TB-1/ding_2026/models/mPLUG-Owl3-7B-241101"
MODEL_PATH = os.environ.get(
    "MPLUG_MODEL", _LOCAL if os.path.isdir(_LOCAL) else "mPLUG/mPLUG-Owl3-7B-241101")


def _to_pil(f):
    if isinstance(f, Image.Image):
        return f
    return Image.fromarray(f.astype(np.uint8))


class MplugOwl3:
    def __init__(self, model_path=MODEL_PATH, device="cuda"):
        self.config = AutoConfig.from_pretrained(model_path,
                                                 trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.processor = self.model.init_processor(self.tokenizer)
        self.device = device

    @torch.inference_mode()
    def answer_mc(self, frames, query, max_new_tokens=64):
        """frames: list of RGB arrays (one video). query: full MC prompt."""
        pil = [_to_pil(f) for f in frames]
        messages = [
            {"role": "user", "content": f"<|video|>\n{query}"},
            {"role": "assistant", "content": ""},
        ]
        inputs = self.processor(messages, images=None, videos=[pil])
        inputs.to(self.device)
        inputs.update({"tokenizer": self.tokenizer,
                       "max_new_tokens": max_new_tokens, "decode_text": True})
        g = self.model.generate(**inputs)
        out = g[0] if isinstance(g, (list, tuple)) else g
        return str(out).strip()

    # snapshot/uniform don't need backbone-side selection (gpt-mini does it);
    # provide stubs so run_vmme's generic wiring stays happy.
    def select_frames(self, frames, question, k):
        import numpy as _np
        return sorted(_np.linspace(0, len(frames) - 1, min(k, len(frames)),
                                   dtype=int).tolist())

    def caption(self, frame, question):
        return {"observation": "", "objects": [], "action": ""}
