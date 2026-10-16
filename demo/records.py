"""Demo recordings and where they live. Only numpy/pandas: the hosted viewer imports this
without sinar, TensorFlow, torch or ultralytics (those are only needed to record, see pipeline.py)."""
import os
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# clips, recordings (*.demo.pkl) and rendered videos (*.detect.mp4, *.track.mp4)
VID_DIR = Path(os.environ.get("SINAR_DEMO_VID_DIR") or Path(__file__).resolve().parent.parent / "vid")
VIDEO_KINDS = ("detect", "track")  # annotated videos for steps 1 and 2
TRAIL = 60  # frames of trail drawn behind each track
FORMAT = 2  # recording layout version; 2 = greedy mode


@dataclass
class VideoInfo:
    """sinar.utils.VideoInfo without the cv2/TensorFlow imports"""
    width: int
    height: int
    fps: int
    total_frames: Optional[int] = None

    @property
    def duration(self) -> Optional[float]:
        if self.fps and self.total_frames:
            return self.total_frames / self.fps
        return None


@dataclass
class FrameRec:
    frame_idx: int
    xyxy: np.ndarray            # (n, 4) pixels
    conf: np.ndarray            # (n,)
    ids: Optional[np.ndarray]   # (n,) int, None when the tracker gave no IDs
    xyn: np.ndarray             # (n, 2) normalized centroids (get_xyid)


@dataclass
class WindowRec:
    index: int                  # greedy mode: window i starts at frame i
    sampled_frames: list[int]   # source frame of each matrix row (<30 for windows cut by the clip end)
    matrix: pd.DataFrame        # 30xN raw window matrix, columns x{id}, y{id}
    cnn_input: np.ndarray       # 30x30, what the CNN actually sees
    prob: float
    is_gang: bool

    @property
    def start(self) -> int:
        return self.sampled_frames[0]

    @property
    def end(self) -> int:
        return self.sampled_frames[-1]


@dataclass
class DemoRun:
    clip: str
    video_info: VideoInfo
    threshold: float
    frames: list[FrameRec] = field(default_factory=list)
    windows: list[WindowRec] = field(default_factory=list)
    mean_prob: float = 0.0      # greedy verdict: mean of all window probabilities >= threshold
    is_gang: bool = False
    elapsed: float = 0.0
    version: int = FORMAT

    # plain dicts on disk: loading needs only numpy/pandas, not these classes' module
    def save(self, path: Path):
        data = asdict(self)
        data["video_info"] = asdict(self.video_info)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "DemoRun":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != FORMAT:
            raise ValueError(f"{path} is an old recording, process the clip again")
        return cls(
            clip=data["clip"],
            video_info=VideoInfo(**data["video_info"]),
            threshold=data["threshold"],
            frames=[FrameRec(**fr) for fr in data["frames"]],
            windows=[WindowRec(**w) for w in data["windows"]],
            mean_prob=data["mean_prob"],
            is_gang=data["is_gang"],
            elapsed=data["elapsed"],
        )


def cache_path(clip) -> Path:
    clip = Path(clip)
    return clip.with_name(f"{clip.stem}.demo.pkl")


def video_paths(clip) -> dict[str, Path]:
    clip = Path(clip)
    return {kind: clip.with_name(f"{clip.stem}.{kind}.mp4") for kind in VIDEO_KINDS}


def is_rendered(path) -> bool:
    return any(Path(path).name.endswith(f".{kind}.mp4") for kind in VIDEO_KINDS)
