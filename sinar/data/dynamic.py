
from collections import defaultdict
import os
from typing import Union
import cv2
import pandas as pd
import torch
from sinar.utils import get_xyid, to_dict
from torch.utils.data import Dataset, DataLoader, Sampler
from ultralytics import YOLO

class CentroidMatrixDataset(Dataset):
    def __init__(self, model_path: str, data=None):
        if model_path is not None:
            self.yolo = YOLO(model_path, task="detect")
        # Group data by their last dimension
        self.grouped_data = data if data is not None else defaultdict(list)
        self.grouped_list = []
        self._rebuild_index()

    def _rebuild_index(self):
        self.grouped_list = [
            (key, idx)
            for key in self.grouped_data.keys()
            for idx in range(len(self.grouped_data[key]))
        ]

    def __len__(self):
        return len(self.grouped_list)

    def __getitem__(self, idx):
        group_key, item_idx = self.grouped_list[idx]
        matrix, label = self.grouped_data[group_key][item_idx]
        return matrix, label

    def create_matrices_from_videos(self, video_path: str, label: int):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = self.yolo.track(frame, verbose=False,
                            stream_buffer=True,
                            persist=True, vid_stride=True,
                            tracker="bytetrack.yaml")[0]
            if res.boxes.id is None:
                rows.append({})
                continue
            ids, xy = get_xyid(res.boxes)
            rows.append(to_dict(ids, xy, flatten=True))
        cap.release()
        self.yolo.predictor.trackers[0].reset() # Reset the tracker to make sure it doesn't hold state from previous video
        # generate matrices
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
                matrix = df.values
                matrices.append(matrix)
            if start_frame + (30 - 1) * 5 >= total_frames:
                break


        # Group matrices by their last dimension (jumlah motor x2)
        for matrix in matrices:
            # skip if empty
            if matrix.shape[-1] == 0:
                print(f"[WARNING] {video_path} contains no motor, skipping")
                break
            # add channel dim (30, x) => (1, 30, x)
            matrix = matrix.reshape(1, -1, matrix.shape[-1])
            # convert to torch tensor
            matrix = torch.from_numpy(matrix)
            # label = torch.tensor(label)
            # add to group
            self.grouped_data[matrix.shape[-1]].append((matrix, label))

        self._rebuild_index()

    def create_from_directory(self, dir_path: str):
        """Process all videos in a directory.
        label_fn(video_filename) -> int
        """
        for fname in os.listdir(dir_path):
            if not fname.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            label = 1 if "gang" in fname.lower() else 0
            print(f"Processing {fname} with label {label}")
            self.create_matrices_from_videos(os.path.join(dir_path, fname), label)

    def save(self, path: str):
        # make sure the data was generated
        if not self.grouped_data:
            raise ValueError("No data available to save.")

        torch.save(dict(self.grouped_data), path)

    @classmethod
    def load(cls, path: str):
        data = torch.load(path)
        return cls(model_path=None, data=data)


# dummy random dataset class for CentroidMatrixDataset
class GroupedDataset(Dataset):
    def __init__(self, grouped_data: list, labels: list):
        self.grouped_data = defaultdict(list)
        for matrix, label in zip(grouped_data, labels):
            self.grouped_data[matrix.shape[-1]].append((matrix, label))
        self.grouped_list = [
            (key, idx) for key in self.grouped_data.keys() for idx in range(len(self.grouped_data[key]))
        ]

    def __len__(self):
        return len(self.grouped_list)

    def __getitem__(self, idx):
        group_key, item_idx = self.grouped_list[idx]
        matrix, label = self.grouped_data[group_key][item_idx]
        return matrix, label   # ✅ return the real label only


class GroupedBatchSampler(Sampler):
    def __init__(self, dataset: Union[CentroidMatrixDataset, GroupedDataset], batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_indices = defaultdict(list)
        # for idx, (_, group_key) in enumerate(dataset):
        #     self.group_indices[group_key].append(idx)
        for idx in range(len(dataset)):
            group_key = dataset.grouped_list[idx][0]  # <-- use dataset metadata
            self.group_indices[group_key].append(idx)

    def __iter__(self):
        for group_key, indices in self.group_indices.items():
            # Yield batches from each group
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size
                   for indices in self.group_indices.values())

