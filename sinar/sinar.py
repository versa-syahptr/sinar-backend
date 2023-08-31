from ultralytics import YOLO
from multiprocessing import Event
import time
from supervision.video.dataclasses import VideoInfo

from .stream import RTMPStream, YTSTREAM
from .predigenk import Anbev
from .utils import cvtext, Process_wrapper, check_stream
from .logger import logger
import cv2

MAXSHAPE = 30
SAMPLING = 5
STEP = 1

# logger = get(__name__)

class SINAR:
    def __init__(self, yolo_model, abModel):
        self.yolo_model = YOLO(yolo_model)
        # self.yolo_model.to(0)
        self.yolo_model.fuse()
        
        # self.device = device
        logger.info(f"yolo model loaded [{yolo_model}]")
        self.ab_predictor = Anbev(abModel, threaded=False)
    
    def __call__(self, source,
                 streamto: RTMPStream = None, 
                 frame_preprocessor=None, 
                 stop_event: Event = None):

        # check stream availability
        retry_count = 0
        while not check_stream(source):
            logger.info(f"({retry_count}) stream {source} is offline, retrying...")
            time.sleep(5)
            retry_count += 1
        
        cam_id = source.split("/")[-1]
        result_generator = self.yolo_model.track(source, device=0, stream=True, verbose=False)
        logger.info(f"tracker start ({source})")

        for result in result_generator:
            frame = result.plot()
            logger.debug(f"{result.verbose()}; speed: {sum(result.speed.values()):.2f}ms")

            # put result to analysis behavior predictor
            self.ab_predictor.put_result(result.cpu())

            # predict
            if self.ab_predictor.ready():
                pred = self.ab_predictor.predict()
                if pred:
                    frame = cvtext(frame, "ADA GENG MOTOR")

            if frame_preprocessor is not None:
                frame = frame_preprocessor(frame)

            # write to stream
            if streamto is not None:
                streamto.write(frame)
            # stop event
            if stop_event is not None and stop_event.is_set():
                break
        logger.info("tracker stop")
        # stop analysis behavior predictor
        # self.ab_predictor.stop()
        streamto.stop()
        logger.info("stream stopped")
