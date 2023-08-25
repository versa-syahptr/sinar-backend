# imports
import numpy as np
import tensorflow as tf
from threading import Thread, Event
from queue import Queue, Empty

from utils import get_xyid, to_dict, fill_square, make_dataframe
import logger

logger = logger.get(__name__)

# analysis behavior class
class Anbev(Thread):
    def __init__(self, model, **kwargs) -> None:
        super(Anbev, self).__init__()
        self.name = "Analysis Behavior"
        self.model = tf.keras.models.load_model(model)
        logger.info(f"Analysis Behavior model loaded [{model}]")

        self.size = kwargs.get("size", 30)
        self.sampling = kwargs.get("sampling", 5)
        self.step = kwargs.get("step", 0)
        self.centroids = []

        # events
        self.stop_event = Event()
        self.genk_event = Event()
        # queues
        self.result_queue = Queue()
    
    def predict(self, x: np.array):
        pred = self.model(np.array([x]))
        return pred.numpy().round().astype(int)[0,0]

    def run(self):
        logger.info("Analysis Behavior thread started")
        idx_frame = 0
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get_nowait()
            except Empty:
                continue

            if (idx_frame == 0 or idx_frame%5 == 0) and len(self.centroids) < self.size:
                if result.boxes.id is None:
                    self.centroids.append(dict()) # put empty row
                else:
                    ids, xy = get_xyid(result.boxes)
                    self.centroids.append(to_dict(ids, xy, flatten=True))
                logger.debug(f"frame {idx_frame} captured ✔")
                

            if len(self.centroids) >= self.size: # full
                df = make_dataframe(self.centroids)
                idx_frame = 0
                # PREDICT
                x = fill_square(df.values, self.size)
                logger.info("Analysis Behavior predicting ")
                preds = self.predict(x)
                logger.info(f"\npreds result: {'GENG MOTOR' if preds else 'aman 👌'}\n")
                
                if preds:
                    self.genk_event.set()
                else:
                    self.genk_event.clear()
            idx_frame += 1
        logger.info("Analysis Behavior thread stopped")
    
    def stop(self):
        self.stop_event.set()
        self.join()


