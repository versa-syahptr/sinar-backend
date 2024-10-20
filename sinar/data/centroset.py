import os
import glob
from pathlib import Path
import random
import cv2
import numpy as np
import tensorflow as tf

from sinar.data.centrogen import Centrogen


class Centroset(tf.keras.utils.PyDataset):
    """
    A sinar.data.Centrogen wrapper, 
    usefull for training models with sinar.data.Centrogen augmentation accross epochs.
    """
    def __init__(self, model_path: str,
                 video_path: str, 
                 batch_size: int,
                 device: str = "cpu",
                 # augmentation params
                 augment_max_trans: int = None,
                 augment_max_angle: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.video_path = video_path
        self.batch_size = batch_size
        self.device = device
        self.centrogen = Centrogen(model_path=self.model_path,
                                   device=self.device,
                                   output_shape=(30, 30),
                                   verbose=False,
                                   augment_max_angle=augment_max_angle,
                                   augment_max_trans=augment_max_trans)
        self.total_matrices = self.get_total_matrices()
        print(f"Total matrices: {self.total_matrices}")
        self.current_dataset = self.centrogen.flow_from_directory(self.video_path, self.batch_size, cache=True)
        
    def get_total_matrices(self):
        files = list(Path(self.video_path).glob(glob))
        print(f"Found {len(files)} videos")
        total_matrices = 0
        for file in files:
            cap = cv2.VideoCapture(str(file))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            matrices = 5 * ((total_frames - 145) // 5) + 5 if total_frames > 145 else 5
            # print(f"Video: {file.name} Total frames: {total_frames}, total matrices: {matrices}")
            total_matrices += matrices
            cap.release()
        return total_matrices
    
    def __len__(self):
        # return self.total_matrices in batched form
        return self.total_matrices // self.batch_size + 1
    
    def __getitem__(self, idx):
        pass




class SavedCentroset(tf.keras.utils.PyDataset):
    """
    A PyDataset class representing a dataset of center points.
    usefull for training models with saved sinar.data.Centrogen data.
    """
    def __init__(self, path, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batch_size = batch_size
        self.saves = self.reshuffle()
        print(f"there are {len(self.saves)} saved datasets")
        self._current_tfds = None
        self._current_X_array = None
        self._current_y_array = None

    def reshuffle(self):
        saves = glob.glob(os.path.join(self.path, "*/"))
        random.shuffle(saves)
        return saves

    def _load_and_shuffle(self):
        if len(self.saves) == 0:
            print(f"all datasets used, reshuffling")
            self.saves = self.reshuffle()
        selected_ds = self.saves.pop(0)
        print(f"using {selected_ds}")
        tfds = tf.data.Dataset.load(selected_ds)
        padding_ds = tf.data.Dataset.from_tensors((tf.zeros([30, 30], dtype=tf.float32), tf.zeros([], dtype=tf.int32)))
        num_pad_samples = self.batch_size - len(tfds) % self.batch_size
        tfds = tfds.concatenate(padding_ds.repeat(num_pad_samples))
        tfds = tfds.shuffle(tfds.cardinality())
        tfds = tfds.batch(self.batch_size)
        tfds = tfds.prefetch(tf.data.experimental.AUTOTUNE)
        return tfds

    def __len__(self):
        if self._current_tfds is None:
            self._current_tfds = self._load_and_shuffle()
        return len(self._current_tfds)

    def __getitem__(self, idx):
        if self._current_tfds is None:
            self._current_tfds = self._load_and_shuffle()

        # convert tf.data.Dataset to numpy array
        if self._current_X_array is None:
            ds_x = self._current_tfds.map(lambda x, y: x)
            self._current_X_array = np.array(list(ds_x.as_numpy_iterator()))
            ds_y = self._current_tfds.map(lambda x, y: y)
            self._current_y_array = np.array(list(ds_y.as_numpy_iterator()))

        return self._current_X_array[idx], self._current_y_array[idx]

    def on_epoch_end(self):
        self._current_tfds = None
        self._current_X_array = None
        self._current_y_array = None

    def on_epoch_begin(self):
        self._current_tfds = self._load_and_shuffle()