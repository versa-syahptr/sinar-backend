import numpy as np
import pandas as pd
import cv2

from collections import namedtuple

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
    text_position = (320, 240)  # (x, y) position of the text
    cv2.putText(frame, text, text_position, font, font_scale, text_color, text_thickness)
    return frame

def check_stream(stream):
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        return False
    cap.release()
    return True