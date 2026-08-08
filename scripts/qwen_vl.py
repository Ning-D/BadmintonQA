"""
Qwen2.5-VL-7B wrapper — the shared answering/vision backbone for both the
`uniform` baseline and the static evidence-graph method.

Loaded once; exposes:
  - answer_mc(frames, query)        -> raw text answer (expects an A-D letter)
  - score_frames(frames, question)  -> per-frame relevance score in [0,1]
  - caption(frame, question)        -> short structured caption (evidence node)
  - text(prompt)                    -> text-only call (edge typing / chain pick)

Frames are numpy HxWx3 RGB arrays (as decoded by decord).
"""
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

# Default to Qwen2-VL-7B-Instruct so we can compare against the LongVideoBench
# leaderboard's Qwen2-VL-7B = 56.8 with the SAME model (no version bonus).
# Override with env QWEN_MODEL to validate transfer to a newer backbone
# (e.g. QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct).
MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct")


def _to_pil(frame):
    if isinstance(frame, Image.Image):
        return frame
    return Image.fromarray(frame.astype(np.uint8))


def _extract_json(text):
    """Pull the first {...} or [...] JSON blob out of an LLM reply."""
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


class QwenVL:
    def __init__(self, model_id=MODEL_ID, device="cuda", max_side=512):
        self.processor = AutoProcessor.from_pretrained(model_id)
        if ("3-VL" in model_id or "Qwen3VL" in model_id) and Qwen3VLForConditionalGeneration is not None:
            cls = Qwen3VLForConditionalGeneration
        elif "2.5" in model_id:
            cls = Qwen2_5_VLForConditionalGeneration
        else:
            cls = Qwen2VLForConditionalGeneration
        self.model = cls.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa",
        ).eval()
        self.max_side = max_side

    # ── low level ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _generate(self, messages, max_new_tokens=256):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images = [c["image"] for m in messages for c in m["content"]
                  if isinstance(c, dict) and c.get("type") == "image"]
        inputs = self.processor(
            text=[text], images=images or None, padding=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

    def _msg(self, frames, prompt, labels=None, raw_labels=False):
        content = []
        for i, f in enumerate(frames):
            if labels and i < len(labels):     # ground each frame with its timestamp
                # raw_labels: emit the label verbatim (e.g. Qwen3-VL's native
                # "<12.4 seconds>" temporal marker); else wrap in [ ] (legacy).
                txt = labels[i] if raw_labels else f"[{labels[i]}]"
                content.append({"type": "text", "text": txt})
            content.append({"type": "image", "image": _to_pil(f)})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ── public API ───────────────────────────────────────────────────────────
    def answer_mc(self, frames, query, max_new_tokens=64, labels=None, raw_labels=False):
        """frames: list of RGB arrays. query: full MC prompt w/ options."""
        return self._generate(self._msg(frames, query, labels, raw_labels), max_new_tokens)

    @torch.inference_mode()
    def answer_mc_conf(self, frames, query, max_new_tokens=8, labels=None, nopt=5):
        """Like answer_mc but also returns the model's CONFIDENCE in the answer,
        read at the step that actually emits the option LETTER (not blindly the
        first token -- the model may emit a space/'('/'The answer is' first).
        Returns (text, conf) where conf has:
          p_top   = softmax prob of the emitted letter token at that step,
          margin  = p_top - p_2nd over the FULL vocab,
          p_letter= prob restricted to the option-letter tokens {A..}, i.e. the
                    model's distribution OVER OPTIONS (calibrated answer conf),
          letter  = the chosen letter, found = whether a letter step was found."""
        import torch.nn.functional as F
        messages = self._msg(frames, query, labels)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images = [c["image"] for m in messages for c in m["content"]
                  if isinstance(c, dict) and c.get("type") == "image"]
        inputs = self.processor(text=[text], images=images or None, padding=True,
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, output_scores=True,
                                  return_dict_in_generate=True)
        gen = out.sequences[0, inputs.input_ids.shape[1]:]
        tok = self.processor.tokenizer
        txt = tok.decode(gen, skip_special_tokens=True).strip()
        # locate the FIRST generated token that decodes to an A-G letter
        letters = [chr(ord("A") + i) for i in range(max(2, min(nopt, 7)))]
        pos, letter = None, None
        for i, tid in enumerate(gen.tolist()):
            piece = tok.decode([tid]).strip().upper()
            if piece[:1] in letters:
                pos, letter = i, piece[:1]; break
        if pos is None or pos >= len(out.scores):
            return txt, {"p_top": float("nan"), "margin": float("nan"),
                         "p_letter": float("nan"), "letter": None, "found": False,
                         "p_options": []}
        probs = F.softmax(out.scores[pos][0].float(), dim=-1)
        top2 = torch.topk(probs, 2)
        # restricted distribution over the option-letter tokens (first token id of each letter)
        lids = [tok.encode(L, add_special_tokens=False)[:1] for L in letters]
        lids = [x[0] for x in lids if x]
        lp = probs[lids]
        p_letter = float(probs[gen[pos]] / lp.sum()) if lp.sum() > 0 else float("nan")
        # full normalized distribution over the option letters A..(nopt) -- the model's
        # calibrated belief per option, gold-free; p_options[k] = belief in option k.
        p_options = [float(probs[i] / lp.sum()) for i in lids] if lp.sum() > 0 else []
        return txt, {"p_top": float(top2.values[0]),
                     "margin": float(top2.values[0] - top2.values[1]),
                     "p_letter": p_letter, "letter": letter, "found": True,
                     "p_options": p_options}

    def answer_mc_grouped(self, query, groups, max_new_tokens=64):
        """Structured/interleaved evidence: groups = [(label_text, [frames])...].
        Builds [question] + per-group [label][its frames] + [answer instruction],
        so the model sees which frames serve which sub-question (visual structure,
        no verbal reasoning needed)."""
        content = [{"type": "text", "text": query}]
        for label, frames in groups:
            content.append({"type": "text", "text": label})
            for f in frames:
                content.append({"type": "image", "image": _to_pil(f)})
        content.append({"type": "text",
                        "text": "Answer with the option's letter only."})
        return self._generate([{"role": "user", "content": content}], max_new_tokens)

    def score_frames(self, frames, question):
        """Return a relevance score in [0,1] for each frame (batched call)."""
        prompt = (
            f"You are given {len(frames)} video frames in temporal order. "
            f"Rank how useful EACH frame is for answering the question, "
            f"RELATIVE to the others — spread the scores out so the most "
            f"useful frames score near 1.0 and the least useful near 0.0 "
            f"(do not give everything the same score).\n"
            f"Question: {question}\n"
            f'Return JSON only: a list of {len(frames)} floats in [0,1], one '
            f"per frame in order, e.g. [0.1, 0.8, ...]."
        )
        raw = self._generate(self._msg(frames, prompt), max_new_tokens=256)
        obj = _extract_json(raw)
        # Model may return {"scores": [...]} OR a bare [...] list.
        if isinstance(obj, dict):
            scores = obj.get("scores")
        elif isinstance(obj, list):
            scores = obj
        else:
            scores = None
        if not isinstance(scores, list) or len(scores) != len(frames):
            return [0.5] * len(frames)  # fallback: neutral
        out = []
        for s in scores:
            try:
                out.append(max(0.0, min(1.0, float(s))))
            except (TypeError, ValueError):
                out.append(0.5)
        return out

    def select_frames(self, frames, question, k):
        """Ask the model to directly pick the k most useful frame indices for
        the question (more discriminative than per-frame scoring). Returns a
        sorted list of k indices; falls back to evenly-spaced on parse failure.
        """
        n = len(frames)
        k = min(k, n)
        prompt = (
            f"You are given {n} video frames in temporal order, indexed 0 to "
            f"{n-1}. Choose the {k} frames that are MOST useful for answering "
            f"the question below — pick the ones that actually show the people, "
            f"objects, actions or moments the question is about.\n"
            f"Question: {question}\n"
            f"Return JSON only: a list of exactly {k} distinct integer indices "
            f"(0-{n-1}), most useful first, e.g. [3, 7, 12]."
        )
        raw = self._generate(self._msg(frames, prompt), max_new_tokens=128)
        obj = _extract_json(raw)
        idx = obj if isinstance(obj, list) else None
        clean = []
        if idx:
            for v in idx:
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= iv < n and iv not in clean:
                    clean.append(iv)
        if len(clean) < k:  # pad/fallback with evenly-spaced frames
            import numpy as _np
            for iv in _np.linspace(0, n - 1, k, dtype=int).tolist():
                if iv not in clean:
                    clean.append(iv)
                if len(clean) >= k:
                    break
        return sorted(clean[:k])

    def caption(self, frame, question):
        """Short evidence-node caption for one core frame."""
        prompt = (
            "Describe this frame as evidence for the question.\n"
            f"Question: {question}\n"
            'Return JSON only: {"observation": "2 sentences", '
            '"objects": ["..."], "action": "1 sentence or none"}'
        )
        raw = self._generate(self._msg([frame], prompt), max_new_tokens=160)
        obj = _extract_json(raw)
        if isinstance(obj, dict):
            return obj
        return {"observation": raw[:200], "objects": [], "action": ""}

    def option_support(self, frames, question, options, max_new_tokens=80):
        """Vision option-support judge (drop-in for the gpt-5.4-mini SYSTEM_OPTJUDGE
        used at graph-build time): LOOK at a moment's frames and score how it bears on
        EACH multiple-choice option, -2..+2. Returns a list of len(options) floats
        (zeros on parse failure). Used to re-bake node['option_support'] with Qwen3-VL
        instead of gpt. Mirrors SYSTEM_OPTJUDGE's rubric and A,B,C order."""
        L = "ABCDEFG"
        opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(options))
        prompt = (
            "You see a few frames from ONE moment of a video, plus a multiple-choice "
            "question and its lettered options. By LOOKING at the frames, judge how THIS "
            "moment bears on each option being the correct answer: +2 strong visual "
            "evidence FOR, +1 weak, 0 irrelevant, -1 weak against, -2 strong against. "
            "Read fine attributes (clothing color, headgear, hair, who/what appears).\n"
            f"Question: {question}\nOptions:\n{opts}\n"
            'Return ONLY JSON: {"support":[a,b,...]} one int per option in A,B,C,... order.')
        raw = self._generate(self._msg(frames, prompt), max_new_tokens)
        obj = _extract_json(raw)
        sup = obj.get("support") if isinstance(obj, dict) else (obj if isinstance(obj, list) else None)
        if not isinstance(sup, list):
            return [0.0] * len(options)
        out = []
        for v in sup[:len(options)]:
            try:
                out.append(max(-2.0, min(2.0, float(v))))
            except (TypeError, ValueError):
                out.append(0.0)
        return (out + [0.0] * len(options))[:len(options)]

    def text(self, prompt, max_new_tokens=256):
        msg = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._generate(msg, max_new_tokens)
