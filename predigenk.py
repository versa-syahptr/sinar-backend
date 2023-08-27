# imports
import numpy as np
import tensorflow as tf
from threading import Thread, Event
from queue import Queue, Empty

from utils import get_xyid, to_dict, fill_square, make_dataframe
import logger

logger = logger.get(__name__, level="DEBUG")

# analysis behavior class
class Anbev():
    def __init__(self, model, threaded=False,**kwargs) -> None:
        super(Anbev, self).__init__()
        self.name = "Analysis Behavior"
        self.model = tf.keras.models.load_model(model)
        logger.info(f"Analysis Behavior model loaded [{model}]")

        self.size = kwargs.get("size", 30)
        self.sampling = kwargs.get("sampling", 5)
        self.step = kwargs.get("step", 0)
        self.centroids = []
        self.idx_frame = 0

        # events
        self.stop_event = Event()
        self.genk_event = Event()
        # queues
        self.result_queue = Queue()
        if threaded:
            self.thread = Thread(target=self.run, name=self.name)
        else:
            self.thread = None

    def start(self):
        if self.thread is not None:
            self.thread.start()
            return self
    
    def ready(self):
        return len(self.centroids) >= self.size

    def predict(self):
        self.idx_frame = 0
        df = make_dataframe(self.centroids)
        # PREDICT
        x = fill_square(df.values, self.size)
        logger.info("Analysis Behavior predicting ")
        with tf.device("cpu"):
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

    def run(self):
        logger.info("Analysis Behavior thread started")
        idx_frame = 0
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get_nowait()
            except Empty:
                continue
            
            self.put_result(result)                

            if self.ready(): # full
                preds = self.predict()
                if preds:
                    self.genk_event.set()
                else:
                    self.genk_event.clear()

        logger.info("Analysis Behavior thread stopped")
    
    def stop(self):
        self.stop_event.set()
        self.join()


