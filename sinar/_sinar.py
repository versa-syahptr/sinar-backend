from pathlib import Path
from typing import Union, Optional, Literal, TypeVar, Callable

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
from ultralytics.engine.results import Results
from ultralytics.utils.ops import xyxy2xywh
from collections import defaultdict
from multiprocessing.synchronize import Event
import time

from .stream import RTMPStream, BaseStream
from .predigenk import Anbev
from .utils import cvtext, check_stream
from .logger import logger
# from notification_service import send_alert_notification
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
        
        self._tracks = defaultdict(list)
        
        # self.device = device
        logger.info(f"yolo model loaded [{yolo_model}]")
        self.ab_predictor = Anbev(abModel, threaded=False)
        self.geng_detected = False
        self.detected_action : Optional[Callable] = None
    
    def register_detected_action(self, action: Callable):
        self.detected_action = action
    
    def predict_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Predict frame from numpy array
        This method should be called in a loop to get the prediction result
        Useful for custom main loop

        Args:
            frame (np.ndarray): input frame
        
        Returns:
            np.ndarray: annotated frame
        """
        result = self.yolo_model.track(frame, device=self.device, 
                                       verbose=False, stream_buffer=True, 
                                       persist=True, vid_stride=True, 
                                       tracker="bytetrack.yaml")[0]
        frame = self._annotate(result)
        logger.debug(f"{result.verbose()} speed: {sum(result.speed.values()):.2f} ms")

        # put result to analysis behavior predictor
        self.ab_predictor.put_result(result.cpu())

        # predict
        if self.ab_predictor.ready():
            self.geng_detected = self.ab_predictor.predict()
            if self.geng_detected and self.detected_action is not None: # do only once
                self.detected_action()
                
        # do as long as pred is true
        if self.geng_detected:
            frame = cvtext(frame, "ADA GENG MOTOR")

        return frame
    
    def __call__(self, source,
                 streamto: BaseStream = _sentinel_stream, 
                 frame_preprocessor : Optional[Callable] = None, 
                 stop_event: Optional[Event] = None):
        self.main_loop(source, streamto, frame_preprocessor, stop_event)
    
    def main_loop(self, source,
                 streamto: BaseStream = _sentinel_stream, 
                 frame_preprocessor : Optional[Callable] = None, 
                 stop_event: Optional[Event] = None):
        """
        Sinar main loop

        Args:
            source (str): source of the stream, could be a file path or an url
            streamto (BaseStream, optional): stream to write the result. Defaults to _sentinel_stream.
            frame_preprocessor (Optional[Callable], optional): frame preprocessor. Defaults to None.
            stop_event (Optional[Event], optional): stop event. Defaults to None.

        """

        capture = cv2.VideoCapture(source)

        # check stream availability
        retry_count = 0
        while not capture.isOpened():
            logger.info(f"({retry_count}) stream {source} is offline, retrying...")
            time.sleep(5)
            retry_count += 1
        
        cam_id = Path(source).stem

        logger.info(f"tracker start ({source})")
        # for result in result_generator:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame = self.predict_frame(frame)

            if frame_preprocessor is not None:
                frame = frame_preprocessor(frame)

            # write to stream
            is_stop = streamto.write(frame)
            # stop event
            if (stop_event is not None and stop_event.is_set()) or not is_stop:
                break
        # clear tracks
        self._tracks.clear()
        logger.info("tracker stop")
        # stop analysis behavior predictor
        # self.ab_predictor.stop()
        streamto.stop()
        logger.info("stream stopped")
    
    def _annotate(self, res: Results):
        im0 = res.orig_img.copy()
        annotator = Annotator(im0, font_size=8)
        boxes = res.boxes.xyxy.cpu()
        if res.boxes.id is not None:
            track_ids = res.boxes.id.cpu().tolist()
               
            for box, track_id in zip(boxes, track_ids):
                annotator.box_label(box, label=f"{int(track_id)}", color=colors(int(track_id)))
                x, y, w, h = xyxy2xywh(box)
                tracks = self._tracks[track_id]
                tracks.append((x, y))
                annotator.draw_centroid_and_tracks(tracks, color=colors(int(track_id)))
        return annotator.result()
    
    # async def predict_from_websocket(self, ws: WebSocket, 
    #                                  streamto: BaseStream = _sentinel_stream,
    #                                  frame_preprocessor=None,
    #                                  stop_event: Optional[Event] = None):
    #     pred = False
    #     while True:
    #         frame_bytes = await ws.receive_bytes()
    #         frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

    #         result = self.yolo_model.track(frame, device=self.device, 
    #                                        verbose=True, persist=True,
    #                                        vid_stride=True, tracker="bytetrack.yaml")[0]
    #         frame = self._annotate(result)

    #         self.ab_predictor.put_result(result.cpu())

    #         if self.ab_predictor.ready():
    #             pred = self.ab_predictor.predict()

    #         if pred:
    #             frame = cvtext(frame, "ADA GENG MOTOR")

    #         if frame_preprocessor is not None:
    #             frame = frame_preprocessor(frame)
            
    #         streamto.write(frame)

    #         if stop_event is not None and stop_event.is_set():
    #             break
