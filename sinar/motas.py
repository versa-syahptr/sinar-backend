# imports
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ultralytics.engine.results import Results
from typing import Optional, Union, Callable, Literal
# import tensorflow as tf
# import pickle

from sinar.utils import get_xyid, to_dict, fill_square, make_dataframe, flatten_matrix
from sinar.models import BaseClassifierModel
import sinar.logger
# from notification_service import send_alert_notification

logger = sinar.logger.get(__name__)

@dataclass
class MotasResult:
    is_gang : bool
    results : np.ndarray
    confidences : Optional[Union[float, np.ndarray]] = None

    @classmethod
    def from_probs(cls, probs: np.ndarray, threshold: float = 0.5):
        is_gang = bool(probs.mean() >= threshold)
        confidences = probs if len(probs) > 1 else (probs[0] if len(probs) == 1 else None)
        results = (probs >= threshold).astype(np.int32)
        return cls(is_gang=is_gang, results=results, confidences=confidences)


class Motas():
    """Motion Analysis class for detecting geng motor"""
    def __init__(self, model: str, live_stream = False, *, threshold: float = 0.5, device: str = "cpu", **kwargs) -> None:
        self.name = "Anbev"
        self.live_stream = live_stream
        self.threshold = threshold

        self.model = BaseClassifierModel.load(model, device=device)

        logger.info(f"Motion Analysis model loaded with type <{self.model.__class__.__name__}>")

        self.size = kwargs.get("size", 30)
        self.sampling = kwargs.get("sampling", 5)
        self.step = kwargs.get("step", 0)
        self.centroids = []
        self.idx_frame = 0
    
    def ready(self):
        return len(self.centroids) >= self.size

    def predict(self) -> MotasResult:
        if self.live_stream:
            return self.predict_on_demand()
        return self.predict_batch()
        
    def predict_on_demand(self) -> MotasResult:
        if self.model is None:
            return
        self.idx_frame = 0
        df = make_dataframe(self.centroids)
        # PREDICT
        # x = fill_square(df.values, self.size)
        x = df.values.astype(np.float32)
        logger.info(f"shape: {x.shape}")
        # ensure x shape is (1, 30, N)
        x = np.expand_dims(x, axis=0)

        logger.info("Motion Analysis predicting on demand")
        pred = self.model.predict(x)
        logger.info(f"preds result: {'GENG MOTOR' if pred else 'aman 👌'}")

        return MotasResult.from_probs(pred, threshold=self.threshold)

    def predict_batch(self) -> MotasResult:
        matrices = []
        for start_point in range(0, len(self.centroids), self.sampling):
            for start_frame in range(start_point, start_point+self.sampling):
                matrix = []
                for i in range(self.size):
                    frame_idx = start_frame + i * self.sampling
                    if frame_idx >= len(self.centroids):
                        matrix.append({})
                        continue
                    matrix.append(self.centroids[frame_idx])
                df = pd.DataFrame(matrix)
                df.fillna(0, inplace=True)
                logger.debug(f"df columns: {df.columns.to_list()}")
                # matrix = fill_square(df.values, self.size)
                matrices.append(df.values)
            if start_frame + (self.size - 1) * self.sampling >= len(self.centroids): # type: ignore
                break
        logger.debug(f"total matrices: {len(matrices)}")
        logger.debug(f"matrices shape: {[m.shape for m in matrices]}")
        # TODO: this line still causes issue if matrices have different shape
        # e.g. (30, 20), (30, 25), (30, 30)
        # need find solution for this
        # either process each matrix one by one
        # or pad the smaller matrix to the largest shape
        # for now, just use `predict_on_demand` instead of this function (the live_stream mode)
        # x = np.array(matrices, dtype=np.float32)
        logger.info("Motion Analysis predicting in batch")
        # logger.info(f"shape: {x.shape}")

        preds = self.model.predict_batch(matrices)

        logger.info(f"preds result: {preds}")
        logger.info(f"preds mean: {preds.mean()}")
        return MotasResult.from_probs(preds, threshold=self.threshold)

    
    def put_result(self, result: Results):
        if (self.idx_frame == 0 or self.idx_frame%self.sampling == 0) and len(self.centroids) < self.size:
            if result.boxes.id is None:
                self.centroids.append(dict()) # put empty row
            else:
                ids, xy = get_xyid(result.boxes)
                self.centroids.append(to_dict(ids, xy, flatten=True))
            logger.debug(f"frame {self.idx_frame} captured ✔")
        self.idx_frame += 1
    
    def reset(self):
        self.centroids = []
        self.idx_frame = 0
        logger.info("Motion Analysis state reset ✔")


