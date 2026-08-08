"""
Dataset-agnostic frame sampling (OpenCV; decord segfaults with torch here).
Shared by VideoEspresso (ve_data) and Video-MME (vmme_data).
"""
import cv2
import numpy as np

# Cap OpenCV/FFmpeg decode threads: long videos otherwise spawn one decode thread
# per core and, in a long-lived process, thrash on lock contention (GPU idles while
# CPU spins). A small fixed pool decodes long ego clips without oversubscription.
cv2.setNumThreads(4)


def _resize(frame, max_side):
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        frame = cv2.resize(frame, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA)
    return frame


def _sample_indices(video_path, idx, n, max_side, return_ts=False):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if total <= 0:
        cap.release()
        return ([], []) if return_ts else []
    if idx is None:
        idx = np.linspace(0, total - 1, min(n, total), dtype=int).tolist()
    frames, kept = [], []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        frames.append(_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side))
        kept.append(i)
    cap.release()
    if return_ts:
        return frames, [round(i / fps, 2) for i in kept]
    return frames


def sample_uniform(video_path, n=16, max_side=512):
    return _sample_indices(video_path, None, n, max_side)


def sample_dense(video_path, n=32, sec_per_frame=None, max_side=512):
    """Candidate-pool sampling. If sec_per_frame is given, the pool size is
    derived per-video so the spacing is CONSTANT (e.g. 20s/frame = 0.05fps),
    overriding n; otherwise a fixed n frames are spread over the whole video."""
    if sec_per_frame:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        if total > 0:
            n = max(1, round((total / fps) / sec_per_frame))
    return _sample_indices(video_path, None, n, max_side, return_ts=True)
