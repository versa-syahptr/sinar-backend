import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from ultralytics import YOLO
from ultralytics.utils import LOGGER as UL_LOGGER
import logging
import tempfile

from sinar.utils import fill_square, get_xyid, to_dict

def generate_augmentation_parameters():
    # Generate random parameters for flip, translation, and rotation
    do_flip = np.random.rand() > 0.5  # Randomly flip the image
    max_trans = 20  # Max translation in pixels
    tx = np.random.uniform(-max_trans, max_trans)
    ty = np.random.uniform(-max_trans, max_trans)
    angle = np.random.uniform(-30, 30)  # Random rotation angle between -30 and 30 degrees
    
    return do_flip, tx, ty, angle

def apply_augmentations(frame, do_flip, tx, ty, angle):
    # Apply flip
    if do_flip:
        frame = cv2.flip(frame, 1)
    
    # Apply translation
    rows, cols, _ = frame.shape
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    frame = cv2.warpAffine(frame, translation_matrix, (cols, rows))
    
    # Apply rotation
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    frame = cv2.warpAffine(frame, rotation_matrix, (cols, rows))
    
    return frame


class Centrogen:
    def __init__(self, model_path: str, 
                 device: str = "cpu", 
                 do_augmentation: bool = True, 
                 output_shape: tuple = (30, 30),
                 verbose: bool = False):
        
        self.model_path = model_path
        self.device = device
        self.do_augmentation = do_augmentation
        self.output_shape = output_shape
        self.verbose = verbose
        self.dataset = None
    
    def _print(self, *args, **kwargs):
        if self.verbose:
            tf.print(*args, **kwargs)

    def create_matrices_from_videos(self, video_path: str) -> np.ndarray:
        if not self.verbose: # suppress ultralytics logger
            _log_level = UL_LOGGER.level
            UL_LOGGER.setLevel(logging.ERROR)
        # check if model_path and video_path are tensors
        self._print(f"\nprocessing {video_path}")
        if isinstance(video_path, tf.Tensor):
            video_path = video_path.numpy().decode('utf-8')

        yolo = YOLO(self.model_path, task="detect", verbose=self.verbose)
        # augment video
        if self.do_augmentation:
            flip_code, tx, ty, angle = generate_augmentation_parameters()
            self._print(f"Augmentation params: flip: {flip_code}, tx: {tx}, ty: {ty}, angle: {angle}")

            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # usable_frames = total_frames - (total_frames % 150)
        self._print(f"Total frames: {total_frames}") #, using {usable_frames} frames")
        # if usable_frames == 0:
        #     raise ValueError(f"Video {video_path} is too short, fix it!")

        # generate matrices rows
        rows = []
        # for _ in range(usable_frames):
        all_id = set()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.do_augmentation:
                frame = apply_augmentations(frame, flip_code, tx, ty, angle)

            res = yolo.track(frame, verbose=False, 
                            stream_buffer=True, 
                            device=self.device,
                            persist=True, vid_stride=True, 
                            tracker="bytetrack.yaml")[0]
            
            if res.boxes.id is None:
                rows.append({})
                continue

            ids, xy = get_xyid(res.boxes)
            rows.append(to_dict(ids, xy, flatten=True))
            all_id.update(ids.astype(int))
            
        cap.release()
        self._print(f"unique ids ({len(all_id)}) : {all_id}")
        # last_frame_idx = 0
        matrices = []
        # while last_frame_idx < total_frames and total_frames-last_frame_idx > 150:
        for start_point in range(0, total_frames, 5):
            for start_frame in range(start_point, start_point+5):
                matrix = []
                for i in range(30):
                    frame_idx = start_frame + i * 5
                    if frame_idx >= total_frames:
                        matrix.append({})
                        continue
                    matrix.append(rows[frame_idx])
                df = pd.DataFrame(matrix)
                df.fillna(0, inplace=True)
                matrix = fill_square(df.values, self.output_shape[0])
                matrices.append(matrix)
            if start_frame + (30 - 1) * 5 >= total_frames:
                break
            # last_frame_idx = frame_idx + 1
        if not self.verbose: UL_LOGGER.setLevel(_log_level)
        return np.array(matrices, dtype=np.float32)

    def map_files_to_labels(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = tf.strings.split(parts[-1], "-")[0]
        if label == "geng":
            label = 1
        else:
            label = 0
        
        return file_path, tf.cast(label, tf.int32)

    def _create_tensor_matrices(self, video_path):
        matrices = tf.py_function(self.create_matrices_from_videos, [video_path], tf.float32)
        matrices.set_shape([None, *self.output_shape])
        return matrices
    
    def _parse_function(self, video_path, label):
        matrices = self._create_tensor_matrices(video_path)
        # Pair each matrix with the label
        labels = tf.fill([tf.shape(matrices)[0]], label)
        # labeled_matrices = [(matrix, label) for matrix in matrices]
        return matrices, labels
    
    def flow_from_directory(self, directory: str, batch_size: int = 32, glob = "*.mp4", cache: bool = False):
        directory = os.path.join(directory, glob)
        file_ds = tf.data.Dataset.list_files(directory)
        # map files to labels to create (file_path, label) tuples
        labeled_ds = file_ds.map(self.map_files_to_labels)
        # create matrices from videos and pair them with labels
        dataset = labeled_ds.flat_map(lambda video_path, label: 
                                    tf.data.Dataset.from_tensor_slices(self._parse_function(video_path, label)))
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
            if cache: 
                dataset = dataset.cache()
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.dataset = dataset
        return dataset
    
    def regenerate_dataset(self):
        if self.dataset is None:
            raise ValueError("Dataset is not yet created, call flow_from_directory first")
        
        tmpdir = tempfile.mkdtemp()
        self.dataset.save(tmpdir)
        loaded_ds = tf.data.Dataset.load(tmpdir)
        return loaded_ds
        