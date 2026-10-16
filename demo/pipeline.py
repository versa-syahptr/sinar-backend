"""Run SINAR (greedy mode) on a clip once and record what every phase produced, for the demo app.

No Streamlit here. Precompute all demo clips before the meeting (recording + annotated videos):

    uv run python demo/pipeline.py vid/*.mp4
    uv run python demo/pipeline.py --render-only vid/*.mp4   # redraw videos from existing recordings
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # before sinar pulls in tensorflow

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics.utils.plotting import Annotator, colors

import sinar.logger
from sinar import SINAR
from sinar.utils import VideoInfo as SinarVideoInfo, fill_square, get_xyid

from records import (VIDEO_KINDS, TRAIL, DemoRun, FrameRec, VideoInfo, WindowRec,  # noqa: F401
                     cache_path, is_rendered, video_paths)

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
DEFAULT_YOLO = str(BACKEND.parent / "models/paper/yolov9m_openvino_model")
DEFAULT_CNN = str(BACKEND.parent / "models/paper/CNN1-set-D49.keras")


def load_sinar(yolo: str = DEFAULT_YOLO, cnn: str = DEFAULT_CNN, device: str = "intel:gpu") -> SINAR:
    s = SINAR(yolo, cnn, live_stream=False, device=device)
    s.benchmark_mode = True  # skip SINAR's own drawing, the app draws per step
    return s


def reset_sinar(s: SINAR):
    s._tracks.clear()
    s.motas.reset()
    s.geng_detected = False
    s.last_result = None
    # no predictor before the first track() call
    if s.yolo_model.predictor is not None:
        s.yolo_model.predictor.trackers[0].reset()


def run(s: SINAR, clip, progress: Optional[Callable[[int, int], None]] = None) -> DemoRun:
    """Process clip with the real SINAR greedy-mode loop, recording each phase."""
    reset_sinar(s)
    vi = SinarVideoInfo.from_video_path(str(clip))
    rec = DemoRun(clip=str(clip), video_info=VideoInfo(vi.width, vi.height, vi.fps, vi.total_frames),
                  threshold=s.motas.threshold)
    motas = s.motas
    state = {"frame_idx": 0}
    captured: dict[str, list] = {}

    orig_put = motas.put_result
    orig_model_batch = motas.model.predict_batch

    def put_result(result):
        orig_put(result)
        boxes = result.boxes
        if boxes.id is not None:
            ids, xyn = get_xyid(boxes)
            ids = ids.astype(int)
        else:
            ids, xyn = None, boxes.xywhn.cpu().numpy()[:, :2]
        rec.frames.append(FrameRec(
            frame_idx=state["frame_idx"],
            xyxy=boxes.xyxy.cpu().numpy(),
            conf=boxes.conf.cpu().numpy(),
            ids=ids,
            xyn=xyn,
        ))

    def model_predict_batch(matrices):
        # the exact matrices Motas.predict_batch built, and what the CNN said about them
        captured["matrices"] = list(matrices)
        captured["probs"] = np.ravel(orig_model_batch(matrices))
        return captured["probs"]

    motas.put_result, motas.model.predict_batch = put_result, model_predict_batch
    try:
        cap = cv2.VideoCapture(str(clip))
        total = rec.video_info.total_frames or 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            s.predict_frame(frame)
            state["frame_idx"] += 1
            if progress is not None:
                progress(state["frame_idx"], total)
        cap.release()
        # what SINAR.main_loop does at the end in greedy mode
        res = motas.predict()
        rec.elapsed = time.time() - t0
        rec.windows = build_windows(motas, captured["matrices"], captured["probs"])
    finally:
        # drop the instance attributes, back to the class/model methods
        del motas.put_result, motas.model.predict_batch
        reset_sinar(s)
    rec.mean_prob = float(np.mean(captured["probs"])) if len(captured["probs"]) else 0.0
    rec.is_gang = bool(res.is_gang)
    return rec


def build_windows(motas, matrices: list[np.ndarray], probs: np.ndarray) -> list[WindowRec]:
    """Label the matrices of Motas.predict_batch: window i holds frames i, i+5, ... (i + 29*5)."""
    n = len(motas.centroids)
    windows = []
    for i, (values, prob) in enumerate(zip(matrices, probs)):
        frames = [f for f in range(i, i + motas.size * motas.sampling, motas.sampling)]
        # same construction as predict_batch, but keeping the column names
        matrix = pd.DataFrame([motas.centroids[f] if f < n else {} for f in frames]).fillna(0)
        assert np.array_equal(matrix.values, values), f"window {i} doesn't match predict_batch's matrix"
        windows.append(WindowRec(
            index=i,
            sampled_frames=[f for f in frames if f < n],
            matrix=matrix,
            cnn_input=fill_square(np.asarray(values, dtype=np.float32), motas.size),
            prob=float(prob),
            is_gang=bool(prob >= motas.threshold),
        ))
    return windows


# ---------- drawing (BGR in, BGR out) ----------
def draw_detections(frame: np.ndarray, rec: DemoRun, idx: int) -> np.ndarray:
    fr = rec.frames[idx]
    ann = Annotator(frame.copy(), font_size=8)
    for box, conf in zip(fr.xyxy, fr.conf):
        ann.box_label(box, label=f"motor {conf:.2f}", color=colors(0))
    return ann.result()


def draw_tracks(frame: np.ndarray, rec: DemoRun, idx: int) -> np.ndarray:
    fr = rec.frames[idx]
    ann = Annotator(frame.copy(), font_size=8)
    if fr.ids is None:
        return ann.result()
    im = ann.im
    scale = np.array([frame.shape[1], frame.shape[0]])
    trails: dict[int, list] = {}
    for past in rec.frames[max(0, idx - TRAIL):idx + 1]:
        if past.ids is not None:
            for tid, xy in zip(past.ids, past.xyn):
                trails.setdefault(int(tid), []).append(xy * scale)
    for box, tid in zip(fr.xyxy, fr.ids):
        c = colors(int(tid))
        ann.box_label(box, label=f"{int(tid)}", color=c)
        pts = trails.get(int(tid), [])
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(im, tuple(map(int, p0)), tuple(map(int, p1)), c, 1)
        if pts:
            cv2.circle(im, tuple(map(int, pts[-1])), 3, c, -1)
    return ann.result()


def draw_hud(frame: np.ndarray, rec: DemoRun, idx: int) -> np.ndarray:
    """Burn the frame number into the top-left corner (read it off a paused video)."""
    lines = [(f"frame {idx}  t {idx / rec.video_info.fps:.2f}s", (255, 255, 255))]
    scale = max(frame.shape[0] / 720, 0.4)
    font, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(scale * 1.5)), round(8 * scale)
    y = pad
    for text, color in lines:
        (w, h), base = cv2.getTextSize(text, font, 0.6 * scale, thick)
        cv2.rectangle(frame, (pad // 2, y - pad // 2), (pad + w + pad // 2, y + h + base + pad // 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (pad, y + h), font, 0.6 * scale, color, thick, cv2.LINE_AA)
        y += h + base + pad
    return frame


DRAWERS = {"detect": draw_detections, "track": draw_tracks}


def render_videos(rec: DemoRun, progress: Optional[Callable[[int, int], None]] = None) -> dict[str, Path]:
    """Draw the recorded boxes onto the clip and encode browser-playable H.264 videos (no YOLO)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found: needed to encode H.264 (cv2's mp4v doesn't play in browsers)")
    cap = cv2.VideoCapture(rec.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or rec.video_info.fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paths = video_paths(rec.clip)
    tmp = {kind: p.with_suffix(".tmp.mp4") for kind, p in paths.items()}
    procs = {kind: subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(tmp[kind])],
        stdin=subprocess.PIPE) for kind in VIDEO_KINDS}
    try:
        for idx in range(len(rec.frames)):
            ok, frame = cap.read()
            if not ok:
                break
            for kind, proc in procs.items():
                proc.stdin.write(draw_hud(DRAWERS[kind](frame, rec, idx), rec, idx).tobytes())
            if progress is not None:
                progress(idx + 1, len(rec.frames))
    finally:
        cap.release()
        for proc in procs.values():
            proc.stdin.close()
            proc.wait()
    for kind, proc in procs.items():
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}) rendering {paths[kind]}")
        tmp[kind].replace(paths[kind])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+")
    parser.add_argument("-y", "--yolo", default=DEFAULT_YOLO)
    parser.add_argument("-c", "--cnn", default=DEFAULT_CNN)
    parser.add_argument("-d", "--device", default="intel:gpu")
    parser.add_argument("--render-only", action="store_true",
                        help="only redraw the annotated videos from existing recordings, no YOLO")
    args = parser.parse_args()
    clips = [c for c in args.clips if not is_rendered(c)]  # vid/*.mp4 also matches our outputs

    if args.render_only:
        for clip in clips:
            paths = render_videos(DemoRun.load(cache_path(clip)))
            print(f"{clip}: " + ", ".join(str(p) for p in paths.values()), flush=True)
        return

    sinar.logger.set_level("warning")
    s = load_sinar(args.yolo, args.cnn, args.device)
    for clip in clips:
        r = run(s, clip)
        out = cache_path(clip)
        r.save(out)
        render_videos(r)
        dur = r.video_info.duration or 0
        above = sum(w.is_gang for w in r.windows)
        print(f"{clip}: {len(r.frames)} frames in {r.elapsed:.1f}s "
              f"({r.elapsed / dur if dur else 0:.1f}x real time) {len(r.windows)} windows, "
              f"{above} above threshold, mean p={r.mean_prob:.3f} -> {'GANG' if r.is_gang else 'normal'} [{out}]",
              flush=True)


if __name__ == "__main__":
    main()
