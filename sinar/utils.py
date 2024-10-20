import numpy as np
import pandas as pd
import cv2
from typing import Tuple, Union

from collections import namedtuple

import tensorflow as tf
from keras import ops

Process_wrapper = namedtuple("Process_wrapper", "process, stop_event")


def get_xyid(boxes, norm=True):
    if norm:
        xy = boxes.xywhn.cpu().numpy()[:,:2]
    else:
        xy = boxes.xywh.cpu().numpy()[:,:2].astype(int)
    ids = boxes.id.cpu().numpy()
    return ids, xy

def to_dict(ids, xyn, flatten=False):
    zipped = zip(ids.astype(int),xyn)
    if flatten:
        row = {}
        for i, (x, y) in zipped:
            row[f"x{i}"] = x
            row[f"y{i}"] = y
        return row
    return dict(zipped)

def fill_square(arr: np.array, size=30):
    padded = np.zeros((size, size), dtype=arr.dtype)
    padded[:arr.shape[0], :arr.shape[1]] = arr[:, :size]
    return padded

def make_dataframe(centroids: list):
    df = pd.DataFrame(centroids)
    df.fillna(0, inplace=True)
    centroids.clear()
    return df

def cvtext(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    text_color = (0,0, 255)  # White color
    text_thickness = 3
    # set text position in the middle of the frame
    text_position = (int((frame.shape[1] - len(text) * 20) / 2), int(frame.shape[0] / 2))
    cv2.putText(frame, text, text_position, font, font_scale, text_color, text_thickness)
    return frame

def check_stream(stream):
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        return False
    cap.release()
    return True


def flatten_matrix(m: Union[np.array, tf.Tensor]):
    """
    flatten matrix to 1D array, if m is 3D array, flatten each matrix in the batch
    """
    if isinstance(m, tf.Tensor):
        m = m.numpy()
    
    if len(m.shape) == 3:
        batch_flattened = []
        for matrix in m:
            batch_flattened.append(flatten_matrix(matrix))
        return np.array(batch_flattened)
    else:
        flattened = []
        m = m.T # transpose
        for i in range(0, len(m), 2):
            for j in range(len(m[i])):
                flattened.append(m[i][j])
                flattened.append(m[i+1][j])
        return np.array(flattened)


@tf.keras.utils.register_keras_serializable(package='Sinar')
class SpecialFlatten(tf.keras.layers.Layer):
    """
    Apply flatten to 2D matrix, if input is 3D array, flatten each matrix in the batch, 
    for usage in keras model
    """
    def __init__(self, **kwargs):
        super(SpecialFlatten, self).__init__(**kwargs)
        self.trainable = False
    
    def _flatten(self, m):
        flattened = []
        m = ops.transpose(m)
        for i in range(0, ops.shape(m)[0], 2):
            for j in range(ops.shape(m)[1]):
                flattened.append(m[i, j])
                flattened.append(m[i+1, j])
        return ops.stack(flattened)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        if len(inputs.shape) == 3:
            batch_size = input_shape[0]
            batch_flattened = []
            for i in range(batch_size):
                batch_flattened.append(self._flatten(inputs[i]))
            return ops.stack(batch_flattened, axis=0)
        else:
            return self._flatten(inputs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2])