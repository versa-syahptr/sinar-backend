# imports
import numpy as np
import tensorflow as tf

from .utils import get_xyid, to_dict, fill_square, make_dataframe
from .logger import logger
from notification_service import send_alert_notification

# logger = get(__name__)

# analysis behavior class
class Anbev():
    def __init__(self, model,**kwargs) -> None:
        self.name = "Anbev"
        with tf.device("CPU"): # type: ignore
            self.model = tf.keras.models.load_model(model)
        logger.info(f"Analysis Behavior model loaded [{model}]")

        self.size = kwargs.get("size", 30)
        self.sampling = kwargs.get("sampling", 5)
        self.step = kwargs.get("step", 0)
        self.centroids = []
        self.idx_frame = 0
    
    def ready(self):
        return len(self.centroids) >= self.size

    def predict(self):
        if self.model is None:
            return
        self.idx_frame = 0
        df = make_dataframe(self.centroids)
        # PREDICT
        x = fill_square(df.values, self.size)
        logger.info("Analysis Behavior predicting ")
        with tf.device("CPU"): # type: ignore
            pred = self.model(np.array([x])).numpy().round().astype(int)[0,0]
        logger.info(f"preds result: {'GENG MOTOR' if pred else 'aman 👌'}")
        return pred
    
    def put_result(self, result):
        if (self.idx_frame == 0 or self.idx_frame%5 == 0) and len(self.centroids) < self.size:
            if result.boxes.id is None:
                self.centroids.append(dict()) # put empty row
            else:
                ids, xy = get_xyid(result.boxes)
                self.centroids.append(to_dict(ids, xy, flatten=True))
            logger.debug(f"frame {self.idx_frame} captured ✔")
        self.idx_frame += 1


