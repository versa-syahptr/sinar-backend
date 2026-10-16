# SINAR Framework
**_Sistem Inteligen Anti Kejahatan Jalan Raya_**

---
### Implementation of Paper:

**Detecting Motorcycle Crime Gangs in CCTV Video Footage Using YOLOv9 and CNN**

At Proceeding of The 4th International Conference on Intelligent Cybernetics Technology & Applications 2024 (ICICyTA 2024)

[doi: 10.1109/ICICYTA64807.2024.10913459](https://doi.org/10.1109/ICICYTA64807.2024.10913459)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Models](#models)
- [CLI Reference](#cli-reference)
  - [Global options](#global-options)
  - [`sinar predict`](#sinar-predict)
  - [`sinar data annotate`](#sinar-data-annotate)
  - [`sinar data view`](#sinar-data-view)
  - [`sinar demo`](#sinar-demo)
  - [`sinar service`](#sinar-service)
  - [Not yet implemented](#not-yet-implemented)
- [Known Issues](#known-issues)
- [Using SINAR as a Library](#using-sinar-as-a-library)
- [Research Notebooks](#research-notebooks)
- [Project Structure](#project-structure)

---

## How It Works

SINAR detects motorcycle gangs (_geng motor_) in CCTV video in two stages:

```
video frame ──► YOLO detector + ByteTrack ──► centroid of every tracked motorcycle
                                                     │  (sampled every 5th frame)
                                                     ▼
                                        30 × N centroid matrix
                                (30 time steps × x,y of each track ID)
                                                     │
                                                     ▼
                              Motas classifier (CNN) ──► gang / normal
```

1. **Detection & tracking** - a YOLO model detects motorcycles and ByteTrack assigns each one a persistent ID.
2. **Motion analysis (Motas)** - the normalized centroid `(x, y)` of each track is recorded every 5th frame. 30 such samples form a matrix whose columns are `x<id>, y<id>` pairs.
3. **Classification** - a small classifier looks at how the motorcycles move together and outputs the probability that the group is a gang. A probability ≥ the threshold (default `0.5`) is reported as **GENG MOTOR**.

---

## Requirements

| Requirement | Needed for |
|---|---|
| Linux (x86_64) | tested platform |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | environment and dependency management |
| Python 3.12 or 3.13 | installed automatically by uv (TensorFlow has no Python 3.14 support yet) |
| `ffmpeg` on `PATH` | `sinar data annotate`, `sinar demo` videos, RTMP streaming in `sinar service` |
| nginx with the RTMP module | `sinar service` live CCTV streaming (see `nginx-rtmp.conf`) |
| Intel CPU/iGPU (optional) | fast inference through OpenVINO |
| NVIDIA GPU (optional) | CUDA inference, see [GPU environment](#gpu-environment) |

---

## Installation

### 1. Install uv

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and create the environment

```bash
$ git clone https://github.com/versa-syahptr/sinar-backend.git
$ cd sinar-backend
$ uv sync
```

`uv sync` downloads Python 3.12 if needed, creates `.venv/`, and installs from `uv.lock`:

- the core dependencies (Ultralytics, OpenVINO, TensorFlow, PyTorch, OpenCV, ...)
- **CPU-only PyTorch** (the default `cpu` group). YOLO inference on Intel hardware is done through OpenVINO, so CUDA is not needed
- the notebook dependencies (the default `lab` group: Jupyter kernel, matplotlib, ...)

### 3. Optional features

| Feature | Command | Adds |
|---|---|---|
| REST service | `uv sync --extra service` | FastAPI, Uvicorn, Motor (MongoDB), geopy |
| Annotator | `uv sync --extra annotator` | labelImg, label-studio-converter |
| Everything | `uv sync --all-extras` | all of the above |

Skip the notebook dependencies with `--no-group lab`.

### GPU environment

PyTorch comes in two mutually exclusive builds, selected by dependency group:

```bash
$ uv sync                               # CPU build (default)
$ uv sync --no-group cpu --group gpu    # CUDA 12.6 build, for deployment on an NVIDIA GPU
```

Enabling both groups at once is rejected by uv. To target CUDA 13.0 instead, change the `pytorch-cu126` index URL in `pyproject.toml` to `https://download.pytorch.org/whl/cu130` and run `uv lock`.

> On a GPU machine, run commands with `uv run --no-group cpu --group gpu sinar ...` (or activate `.venv` and call `sinar` directly). Otherwise a plain `uv run` re-syncs the environment back to the CPU build.

### 4. Verify

```bash
$ uv run sinar --help
```

All commands below can be run either through `uv run sinar ...` or, after `source .venv/bin/activate` (`.venv/bin/activate.fish` for fish), directly as `sinar ...`.

### Install as a standalone tool

To use only the CLI, without cloning:

```bash
$ uv tool install git+https://github.com/versa-syahptr/sinar-backend.git
```

> This installs the PyPI build of PyTorch (CUDA-enabled on Linux, large download), since dependency groups are not available to `uv tool`.

---

## Models

Model files are **not stored in this repository** (`models/` is git-ignored). SINAR needs two models.

### Detector (`-y / --yolo`)

Anything Ultralytics can load:

| Format | Example | Notes |
|---|---|---|
| PyTorch | `best.pt` | needs PyTorch, slow on CPU |
| OpenVINO | `best_openvino_model/` (a directory) | **recommended on Intel CPU/iGPU** |
| ONNX / TFLite / TorchScript | `best.onnx`, ... | other Ultralytics export formats |

Export a `.pt` model to OpenVINO once (works on CPU):

```bash
$ uv run yolo export model=path/to/best.pt format=openvino
# -> path/to/best_openvino_model/
```

### Motion analysis classifier (`-a / --anbev`)

The loader is chosen by file extension:

| Extension | Loaded as | Expected input |
|---|---|---|
| `.keras`, `.h5` | TensorFlow/Keras model | 30 × 30 matrix (zero padded / cropped) |
| `.pth`, `.pt` | PyTorch `state_dict` of `TorchClassifierModel` (`sinar/models.py`) | 30 × N matrix (dynamic width) |
| `.pkl`, `.pickle` | pickled scikit-learn model (e.g. SVM) | flattened 30 × 30 matrix |

> A `.pth` file must match the `TorchClassifierModel` architecture exactly, otherwise `load_state_dict` fails.

### Paper models

The results reported in the ICICyTA 2024 paper were produced with this pair:

| Role | Model | Original training artifact |
|---|---|---|
| Detector | **YOLOv9m**, trained on the `all-data-fix` dataset, 640 × 640 input, run through OpenVINO | `yolov8/train/yolov9m-all-data-fix/weights/best.pt` → `best_openvino_model/` |
| Classifier | **CNN1-set-D49**: 3 convolution layers (16 filters each) + global average pooling + sigmoid, ≈7k parameters, 30 × 30 input | `runs/documentations/CNN1-set-D49/best.keras` |

They were run in **live mode** (`-l`) with the default threshold of `0.5`. Annotated outputs from the paper period (October 2024) are kept in `results/`.

Both models are also bundled in a single PyTorch checkpoint, `models/sinar_model.pt` (made in the `torch.ipynb` notebook, November 2024):

- `model`: the YOLOv9m detection model, stored in fp16
- `motas_state_dict`: CNN1-set-D49 converted from Keras to PyTorch (`MotasClassifier`, defined in `torch.ipynb`; also saved on its own as `models/motas_classifier.pth`)

The Keras → PyTorch conversion is exact: both PyTorch copies reproduce `best.keras` within 1e-7, with 100% the same decisions. `MotasClassifier` is not part of the `sinar` package yet, though, so the PyTorch copies can't be passed to `-a`. Use the original Keras file.

#### Running the paper models

The original training artifacts are archived outside this repository. Copied into `models/paper/` (next to this repo, git-ignored), they are:

| File | Source in the archive |
|---|---|
| `yolov9m_openvino_model/` | `yolov8/train/yolov9m-all-data-fix/weights/best_openvino_model/` (Ultralytics 8.2.91, 2024-09-11) |
| `yolov9m-all-data-fix.pt`, `args.yaml`, `results.csv` | `yolov8/train/yolov9m-all-data-fix/` |
| `CNN1-set-D49.keras` | `runs/documentations/CNN1-set-D49/best.keras` (Keras 3.4.1, 2024-10-22) |

```bash
$ uv run sinar predict -y ../models/paper/yolov9m_openvino_model \
    -a ../models/paper/CNN1-set-D49.keras -i vid/gang-11D.mp4 -d intel:gpu -l -v
```

Re-run on 2026-10-14 (Intel i5-1235U, Iris Xe iGPU through OpenVINO, about 2.8× slower than real time). Probability of each 30-sample window:

| Clip | Window probabilities | Verdict |
|---|---|---|
| `gang-11D` | 0.997, 0.387, 0.976, 0.555 | gang ✔ |
| `gang-1` | 1.000 | gang ✔ |
| `normal-1.004` | 0.023 | normal ✔ |
| `normal-5.003` | 0.000 | normal ✔ |

In live mode each window is logged as `preds result: GENG MOTOR (p=0.997)` / `aman 👌 (p=0.387)`.

---

## CLI Reference

```
sinar [-L LEVEL] <command> [options]
```

| Command | Description |
|---|---|
| `predict` | run gang detection on a video file or stream |
| `data annotate` | extract frames from a video and auto-annotate them with YOLO |
| `data view` | browse a YOLO dataset with its bounding boxes |
| `demo` | start the step-by-step Streamlit demo |
| `service` | run the REST API service |
| `data centrogen`, `train` | placeholders, not implemented yet |

### Global options

| Option | Default | Description |
|---|---|---|
| `-L, --log-level {trace,debug,info,warning,error,critical}` | `info` | SINAR log level. `debug` shows the matrices sent to the classifier, `trace` also logs every YOLO result. |

Global options must come **before** the command:

```bash
$ sinar -L debug predict ...
```

---

### `sinar predict`

Run the full pipeline (detector → tracker → classifier) on a video.

```
sinar predict -y YOLO -a ANBEV -i INPUT [-d DEVICE] [-o OUTPUT] [-v] [-l]
```

| Option | Required | Default | Description |
|---|---|---|---|
| `-y, --yolo YOLO` | ✔ | | Path to the YOLO model: `.pt` file or OpenVINO model directory. See [Models](#models). |
| `-a, --anbev ANBEV` | ✔ | | Path to the motion analysis classifier (`.keras`, `.h5`, `.pth`, `.pt`, `.pkl`). |
| `-i, --input INPUT` | ✔ | | Video source: a file path or a stream URL readable by OpenCV (`.m3u8`, `rtsp://`, `rtmp://`, ...). |
| `-d, --device DEVICE` | | `cpu` | Inference device, passed to both models (see below). |
| `-o, --output OUTPUT` | | | Save the annotated video to this file (mp4). |
| `-v, --view` | | off | Show the annotated video in a window. Press `q` to stop. |
| `-l, --live` | | off | Live mode: classify every 30 sampled frames while the video plays (see below). |

> See [Known Issues](#known-issues) #1 before relying on `-o`.

`-o` and `-v` can't be combined: if `-o` is given, the result is saved and no window is shown. With neither option, the video is only processed and the result is logged.

#### Prediction modes

| Mode | Flag | Behaviour | Output |
|---|---|---|---|
| **Greedy** (default) | | Intended for **recorded video files**: collects the centroids of every frame while the video plays and classifies once at the end, one 30-sample window starting at every frame (≈ frames − 145 windows). The verdict is gang when the mean probability over all windows reaches the threshold. | The final verdict is logged: `Motion Analysis result: GENG MOTOR` or `aman 👌` |
| **Live** | `-l` | Intended for **live streams** (public CCTV `.m3u8`, `rtmp://`) that never end: classifies every time 30 samples (≈150 frames, 5 s at 30 fps) are collected, then starts a new window. It also works on files, and the paper's results were produced that way. | each window is logged as `preds result: GENG MOTOR (p=…)` or `aman 👌 (p=…)`. `ADA GENG MOTOR` is drawn on the frames while the latest window's verdict is gang |

> **Live stream → RTMP output.** `sinar predict` can only show (`-v`) or save (`-o`) the result. Streaming the annotated video back out to an RTMP server (nginx-rtmp, see `nginx-rtmp.conf`) is done by the root `main.py` script (multi-stream, one process per source, e.g. `helper.sh`) and by `sinar service` `/start-tracker`. Both currently fail, see [Known Issues](#known-issues) #3 and #4:
>
> ```bash
> $ python main.py -y best.pt -a classifier.keras \
>     -s https://example.com/cctv/cam1.m3u8 -o rtmp://0.0.0.0/output/cam1 \
>     -s https://example.com/cctv/cam2.m3u8 -o rtmp://0.0.0.0/output/cam2
> ```

The annotated frames show each track's bounding box, its ID, and the trail of its centroid.

#### Devices

| Value | Effect |
|---|---|
| `cpu` | CPU for both models |
| `0`, `cuda:0` | NVIDIA GPU (needs the [GPU environment](#gpu-environment)) |
| `intel:gpu`, `intel:cpu` | OpenVINO device for an OpenVINO YOLO model (e.g. Intel Iris Xe iGPU). The PyTorch classifier doesn't understand this value, logs an error and falls back to CPU, which is harmless since the classifier is tiny. |

#### Examples

```bash
# view the result while processing, OpenVINO YOLO on the Intel iGPU
$ sinar predict -y models/best_openvino_model -a models/classifier.keras \
    -i vid/gang-11D.mp4 -d intel:gpu -v

# live mode on a recorded file, save the annotated video
$ sinar predict -y models/best_openvino_model -a models/classifier.keras \
    -i vid/gang-11D.mp4 -l -o out/gang-11D.mp4

# public CCTV HLS stream (the intended use of live mode), watch it in a window
$ sinar predict -y models/best_openvino_model -a models/classifier.keras \
    -i https://example.com/cctv/stream.m3u8 -l -v

# see the matrices sent to the classifier
$ sinar -L debug predict -y models/best.pt -a models/classifier.pth -i vid/normal-1.004.mp4
```

---

### `sinar data annotate`

Prepare a detection dataset from a video. Requires `ffmpeg` and the `annotator` extra.

```
sinar data annotate -m MODEL -d DATASET_PATH -v VIDEO_PATH [-fps FPS]
```

| Option | Required | Default | Description |
|---|---|---|---|
| `-m, --model MODEL` | ✔ | | YOLO model used to pre-annotate the frames. |
| `-d, --dataset-path DATASET_PATH` | ✔ | | Root directory of the dataset. |
| `-v, --video-path VIDEO_PATH` | ✔* | | Video to process. A relative path is resolved against `DATASET_PATH`. |
| `-fps FPS` | | `1` | Number of frames to extract per second of video. |

\* Not marked as required by the parser, but the command fails without it.

The command:

1. extracts frames with `ffmpeg` into `DATASET_PATH/<video name>/images/`
2. runs YOLO on each frame and writes YOLO-format labels to `DATASET_PATH/<video name>/labels/`, plus `classes.txt`
3. converts the result to a Label Studio predictions file, `DATASET_PATH/<video name>/<video name>.json`, so it can be imported into Label Studio and corrected by hand. The image URLs point to `/data/local-files/?d=dataset/<video name>/images/`, so Label Studio's local file storage must serve the dataset from a `dataset/` directory.

```bash
$ sinar data annotate -m models/best.pt -d ~/dataset -v videos/geng-4.mp4 -fps 2
```

---

### `sinar data view`

Browse the images of a YOLO-format dataset with their bounding boxes drawn on.

```
sinar data view DATASET_DIR
```

| Argument | Description |
|---|---|
| `DATASET_DIR` | Directory containing `images/*.jpg` and `labels/*.txt` (e.g. a folder created by `data annotate`). |

Press `space` (or any other key) for the next image and `q` to quit.

```bash
$ sinar data view ~/dataset/geng-4
```

---

### `sinar demo`

Start the step-by-step demo (detection → tracking → centroid matrix → classification) in the browser. It works from any working directory and needs the source checkout and the `demo` dependency group (installed by default with `uv sync`).

```
sinar demo [--port PORT] [streamlit options...]
```

| Option | Default | Description |
|---|---|---|
| `--port PORT` | `8501` | Port to serve the demo on. |
| other options | | Passed to `streamlit run`, e.g. `--server.headless true`. |

The demo plays recordings, it doesn't run the models live. Record the clips in `vid/` before presenting. This runs SINAR in greedy mode and renders the annotated videos for steps 1–2:

```bash
uv run python demo/pipeline.py vid/*.mp4
uv run python demo/pipeline.py --render-only vid/*.mp4   # only redraw the videos
```

A clip without a recording can also be processed from the app's sidebar.

### `sinar service`

> **Legacy.** The service is the backend for the original SINAR competition project (mobile app + CCTV dashboard). It depends on a MongoDB database and a Google Geocoding API key that are no longer active, so it does not work as-is.

Run the FastAPI REST service. Requires the `service` extra.

```
sinar service [--port PORT]
```

| Option / variable | Default | Description |
|---|---|---|
| `--port PORT` | `8080` | Port to listen on. |
| `PORT` env var | | Overrides `--port`. |
| `HOST` env var | `127.0.0.1` | Interface to bind to. |
| `API_KEY` env var | | Google Geocoding API key used for reverse-geocoding CCTV locations. |

Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/ping` | health check (does a test reverse-geocode) |
| GET | `/list-cctv` | list all CCTVs |
| GET | `/cctv/{cctv_id}` | get one CCTV |
| GET | `/total-crime` | number of CCTVs with status `DANGER` |
| POST | `/insert-cctv` | add a CCTV (JSON body with `lat`, `lng`, ...) |
| POST | `/update-cctv/{cctv_id}` | update a CCTV |
| POST | `/insert-report` | submit a citizen report (form: `name`, `title`, `address`, `date`, `image`) |
| GET | `/list-report` | list reports |
| GET | `/start-tracker/{cctv_id}` | start a SINAR process reading `rtmp://localhost/input/{cctv_id}` and publishing to `rtmp://localhost/output/{cctv_id}` |
| GET | `/stop-tracker/{cctv_id}` | stop that process |

The tracker endpoints expect `best.pt` and `anbev-cnn-fix.keras` in the working directory and an nginx RTMP server configured as in `nginx-rtmp.conf`.

---

### Not yet implemented

`sinar data centrogen` and `sinar train` exist in the CLI but raise `NotImplementedError`. Training is currently done from the notebooks, using `sinar.data.centrogen.Centrogen` (TensorFlow, fixed 30 × 30 matrices), `sinar.data.dynamic.CentroidMatrixDataset` (PyTorch, dynamic 30 × N matrices) and `sinar.trainer.train_loop`.

---

## Known Issues

Found by code review and not fixed yet. The CLI reference above describes the intended behaviour.

| # | Affects | Problem |
|---|---|---|
| 1 | `predict -o` | Crashes with a `TypeError`: `sinar/main.py` calls `stream.Saver(width, height, fps, ...)`, but `Saver` now takes a `VideoInfo`. **Workaround:** use the [library API](#using-sinar-as-a-library). |
| 2 | `predict -d` with a `.pth` classifier | `TorchClassifierModel` moves itself to the device before its layers are created, so the weights stay on the CPU. A CUDA device causes a device mismatch error. |
| 3 | stream input | If the source can't be opened, `main_loop` retries forever without re-opening it. |
| 4 | live streaming to RTMP (`main.py`, `service`) | `RTMPStream.write()` returns `None`, which the main loop treats as "stop", so streaming ends after one frame. In addition, `SINAR.__call__` passes positional arguments to the keyword-only `main_loop`, and both the root `main.py` and `server.py` call `sinar.new_sinar_process` (and `main.py` also `sinar.any_process_alive` / `stop_all_processes`), which aren't exported from the `sinar` package (they live in `sinar.service`). So neither `main.py` nor `/start-tracker` can stream right now. |
| 5 | `data annotate` | `-v/--video-path` isn't marked as required, but the command fails without it. |
| 6 | old `.keras` classifiers | `.keras` files saved with Keras 2.x (TensorFlow ≤ 2.15, e.g. `anbev-cnn-v2.keras`, `sinar_vgg16.keras`) can't be loaded by Keras 3. Files saved with Keras 3 (e.g. `yolov9m-CNN512.keras`) and legacy `.h5` files work. |

---

## Using SINAR as a Library

```python
from sinar import SINAR, stream
from sinar.utils import VideoInfo

sinar = SINAR("models/best_openvino_model", "models/classifier.keras",
              live_stream=False, device="intel:gpu", probability_threshold=0.5)

vi = VideoInfo.from_video_path("vid/gang-11D.mp4")
out = stream.MultiStream([stream.Viewer("gang-11D"), stream.Saver(vi, output="out/gang-11D.mp4")])

result = sinar.main_loop("vid/gang-11D.mp4", streamto=out)   # greedy mode returns a MotasResult
print(result.is_gang, result.confidences)

# live mode (live_stream=True): call sinar.predict_frame(frame) per frame;
# sinar.last_result holds the MotasResult of the latest 30-sample window

sinar.reset()   # clear tracker and classifier state before the next video
```

Available streams in `sinar.stream`: `BaseStream` (discard), `Viewer` (window), `Saver` (mp4 file), `RTMPStream` (ffmpeg to an RTMP URL), `MultiStream` (several at once).
`main_loop(..., benchmark_mode=True)` skips drawing and streaming, to measure raw inference speed.

---

## Research Notebooks

The research notebooks and the figures for the paper live in the orphan **`lab`** branch. Check it out as a worktree next to this repository:

```bash
$ git worktree add ../sinar-lab lab
```

Then select this project's `.venv` (`sinar-backend/.venv/bin/python`) as the Jupyter kernel. The default `lab` dependency group already includes `ipykernel`.

---

## Project Structure

```
sinar/
├── main.py          # CLI entry point (`sinar` command)
├── core.py          # SINAR class: detection + tracking + main loop
├── motas.py         # Motion analysis: centroid sampling and classification
├── models.py        # Classifier loaders (Keras / PyTorch / scikit-learn / OpenVINO)
├── stream.py        # Output streams: Viewer, Saver, RTMPStream, MultiStream
├── utils.py         # Matrix helpers, VideoInfo, SpecialFlatten Keras layer
├── logger.py        # Colored logger with TRACE level
├── trainer.py       # Custom Keras training loop for Centrogen datasets
├── data/
│   ├── annotator.py # `sinar data annotate`
│   ├── viewer.py    # `sinar data view`
│   ├── centrogen.py # TensorFlow centroid-matrix dataset generator
│   ├── centroset.py # Keras PyDataset variant
│   └── dynamic.py   # PyTorch dynamic-size centroid-matrix dataset
└── service/         # Legacy FastAPI service (competition project)
```
