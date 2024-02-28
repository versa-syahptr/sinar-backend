from typing import Union, Optional, Literal, TypeVar

from ultralytics import YOLO
from multiprocessing.synchronize import Event
import time

from .stream import RTMPStream, BaseStream
from .predigenk import Anbev
from .utils import cvtext, check_stream
from .logger import logger
from notification_service import send_alert_notification
import cv2

MAXSHAPE = 30
SAMPLING = 5
STEP = 1

# sentinel stream for default streamto parameter, disallowing None
_sentinel_stream = BaseStream()


class SINAR:
    def __init__(self, yolo_model, abModel, device: Optional[Union[int, Literal["cpu"]]]=0):
        self.yolo_model = YOLO(yolo_model, task="detect")
        # self.yolo_model.to(0)
        self.device = device
        # self.yolo_model.fuse()
        
        # self.device = device
        logger.info(f"yolo model loaded [{yolo_model}]")
        self.ab_predictor = Anbev(abModel, threaded=False)
    
    def __call__(self, source,
                 streamto: BaseStream = _sentinel_stream, 
                 frame_preprocessor=None, 
                 stop_event: Optional[Event] = None):

        # check stream availability
        retry_count = 0
        while not check_stream(source):
            logger.info(f"({retry_count}) stream {source} is offline, retrying...")
            time.sleep(5)
            retry_count += 1
        
        cam_id = source.split("/")[-1]
        result_generator = self.yolo_model.track(source, device=self.device, stream=True, 
                                                 verbose=True, stream_buffer=True, 
                                                 vid_stride=True, tracker="bytetrack.yaml")
        logger.info(f"tracker start ({source})")
        pred = False
        img_index = 0
        for result in result_generator:
            frame = result.plot()
            logger.debug(f"{result.verbose()}; speed: {sum(result.speed.values()):.2f}ms")

            # put result to analysis behavior predictor
            self.ab_predictor.put_result(result.cpu())

            # predict
            if self.ab_predictor.ready():
                pred = self.ab_predictor.predict()
                if pred: # do only once
                    cv2.imwrite(f"/var/www/image/{img_index}-{cam_id}.jpg", frame)
                    img_index += 1
                    send_alert_notification("ADA GENG MOTOR", "Ada geng motor di depan", cam_id, 
                                            f"http://sinar.versa.my.id/image/{img_index}-{cam_id}.jpg")
                    
            # do as long as pred is true
            if pred:
                frame = cvtext(frame, "ADA GENG MOTOR")

            if frame_preprocessor is not None:
                frame = frame_preprocessor(frame)

            # write to stream
            streamto.write(frame)
            # stop event
            if stop_event is not None and stop_event.is_set():
                break
        logger.info("tracker stop")
        # stop analysis behavior predictor
        # self.ab_predictor.stop()
        streamto.stop()
        logger.info("stream stopped")
