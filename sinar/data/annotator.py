from pathlib import Path
import subprocess
import sys
from sinar.logger import logger

from ultralytics import YOLO

def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True)
        return True
    except:
        return False

def extract_frames(video_path: Path, output_dir: Path, fps: int = 1):
    logger.info(f"[extract_frames] Extracting frames {video_path} to {output_dir} with fps={fps}")
    filename_patern = f"frame-{video_path.stem[0] + video_path.stem.split('-')[-1]}-%03d.jpg"
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:v", "2",
        "-vf", f"fps={fps}",
        str(output_dir / filename_patern)
    ]
    subprocess.run(cmd, check=True)
    logger.info("[extract_frames] Done!")

def auto_annotate(model_path: Path, frame_dir: Path, output_dir: Path):
    logger.info(f"[auto_annotate] Annotating {frame_dir} using {model_path} and save to {output_dir}")
    model = YOLO(model_path, task="detect")
    results = model(frame_dir, stream=True)
    for res in results:
        label_path = output_dir / f"{Path(res.path).stem}.txt"
        res.save_txt(label_path)
        logger.info(f"[auto_annotate] {label_path} saved")
    classes_path = output_dir / "classes.txt"
    with open(classes_path, "w") as f:
        f.write("\n".join(res.names.values()))
    logger.info(f"[auto_annotate] {classes_path} saved")
    logger.info("[auto_annotate] Done!")

def start_labeler(images_dir: Path, labels_dir: Path):
    logger.info(f"[labeler] Starting labeler for {images_dir} and {labels_dir}")
    cmd = [
        "labelImg",
        str(images_dir),
        str(labels_dir / "classes.txt"),
        str(labels_dir)
    ]
    subprocess.run(cmd, check=True)
    logger.info("[labeler] Done!")

def main(args):
    video_path = args.video_path
    fps = args.fps
    model_path = args.model
    logger.info(f"video_path: {video_path}")
    logger.info(f"fps: {fps}")
    logger.info(f"model_path: {model_path}")
    
    # check ffmpeg
    if not check_ffmpeg():
        logger.error("ffmpeg not found, please install ffmpeg")
        return 1
    
    # check labelImg
    try:
        subprocess.run(["labelImg", "-h"], check=True)
    except:
        logger.error("labelImg not found, please install annotator feature")
        return
    
    # create paths
    global_dataset_path = video_path.parents[1]
    dataset_path = global_dataset_path / video_path.stem
    for mode in ("val", "train"):
        image_path = dataset_path / mode / "images"
        label_path = dataset_path / mode / "labels"
        image_path.mkdir(parents=True, exist_ok=True)
        print(image_path, "created")
        label_path.mkdir(parents=True, exist_ok=True)
        print(label_path, "created")
    
    # extract frames
    extract_frames(video_path, image_path, fps)
    # auto annotate
    auto_annotate(model_path, image_path, label_path)
    # start labeler
    start_labeler(image_path, label_path)
    # done
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

