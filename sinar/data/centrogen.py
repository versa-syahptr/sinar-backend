import os
from typing import Tuple, Union
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from ultralytics import YOLO
from ultralytics.utils import LOGGER as UL_LOGGER
import logging
import tempfile

from sinar.utils import fill_square, get_xyid, to_dict, flatten_matrix

def generate_augmentation_parameters(max_trans: int = 20, max_angle: int = 30) -> Tuple[bool, float, float, float]:
    # Generate random parameters for flip, translation, and rotation
    do_flip = np.random.rand() > 0.5  # Randomly flip the image
    tx = np.random.uniform(-max_trans, max_trans)
    ty = np.random.uniform(-max_trans, max_trans)
    angle = np.random.uniform(-max_angle, max_angle)  # Random rotation angle between -30 and 30 degrees
    
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
                #  do_augmentation: bool = True, 
                 output_shape: tuple = (30, 30),
                 verbose: bool = False,
                 # augmentation params
                 augment_max_trans: int = None,
                 augment_max_angle: int = None
                ):
        
        self.model_path = model_path
        self.device = device
        self.do_augmentation = augment_max_angle is not None and augment_max_trans is not None
        self.augment_max_angle = augment_max_angle
        self.augment_max_trans = augment_max_trans
        self.output_shape = output_shape
        self.verbose = verbose
        self.dataset = None
        self._batch_size = None
    
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
            flip_code, tx, ty, angle = generate_augmentation_parameters(self.augment_max_trans, self.augment_max_angle)
            self._print(f"Augmentation params: flip: {flip_code}, tx: {tx}, ty: {ty}, angle: {angle}")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._print(f"Total frames: {total_frames}")

        # generate matrices rows
        rows = []
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
        if not self.verbose: UL_LOGGER.setLevel(_log_level)
        return np.array(matrices, dtype=np.float32)

    def _flatten_matrix(self, matrix):
        flattened = tf.py_function(flatten_matrix, [matrix], tf.float32)
        flattened.set_shape([self.output_shape[0] * self.output_shape[1]])
        return flattened

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
    
    # parse function for dataset using interleave mapping
    def _parse_function(self, video_path, label):
        matrices = self._create_tensor_matrices(video_path)
        ds_matrices = tf.data.Dataset.from_tensor_slices(matrices)
        labels = tf.data.Dataset.from_tensor_slices(tf.fill([tf.shape(matrices)[0]], label))
        return tf.data.Dataset.zip((ds_matrices, labels))
    
    def flow_from_directory(self, directory: str, batch_size: int = 32, glob = "*.mp4", cache: bool = False):
        directory = os.path.join(directory, glob)
        file_ds = tf.data.Dataset.list_files(directory)
        # map files to labels to create (file_path, label) tuples
        labeled_ds = file_ds.map(self.map_files_to_labels)
        # create matrices from videos and pair them with labels
        # dataset = labeled_ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.unbatch()

        # use interleave to make sure the dataset is shuffled properly
        dataset = labeled_ds.interleave(self._parse_function, 
                                        cycle_length=tf.data.experimental.AUTOTUNE, block_length=1,
                                        deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
            self._batch_size = batch_size
            if cache: 
                dataset = dataset.cache()
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.dataset = dataset
        return dataset
    
    def regenerate_dataset(self):
        """Regenerate the dataset with the same batch size
        unpack matrices and labels from the dataset -> zip them -> batch them

        Returns:
            tf.data.Dataset: regenerated dataset with known cardinality
        """
        if self.dataset is None:
            raise ValueError("Dataset is not yet created, call flow_from_directory first")
        
        # unbatch the dataset -> "unzip" the matrices and labels -> zip them -> batch them again
        regenerated_ds = tf.data.Dataset.zip(*map(tf.data.Dataset.from_tensor_slices, map(np.array, 
                                            zip(*self.dataset.unbatch().as_numpy_iterator()))))
        regenerated_ds = regenerated_ds.shuffle(regenerated_ds.cardinality().numpy())
        regenerated_ds = regenerated_ds.batch(self._batch_size)
        return regenerated_ds
    
    def create_flatten_dataset(self, return_numpy = False) -> Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]:
        if self.dataset is None:
            raise ValueError("Dataset is not yet created, call flow_from_directory first")
        
        # check if the dataset is batched
        if self.dataset.element_spec[0].shape[0] is None:
            self.dataset = self.dataset.unbatch()
        
        flatten_ds = self.dataset.map(lambda x, y: (self._flatten_matrix(x), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        if return_numpy:
            features, labels = map(np.array, zip(*list(flatten_ds.as_numpy_iterator())))
            return features, labels
        return flatten_ds