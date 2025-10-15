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
import cv2

from sinar.stream import RTMPStream, BaseStream
from sinar.motas import Motas, MotasResult
from sinar.utils import cvtext, check_stream
import sinar.logger
# from notification_service import send_alert_notification

logger = sinar.logger.get(__name__)

MAXSHAPE = 30
SAMPLING = 5
STEP = 1

# sentinel stream for default streamto parameter, disallowing None
_sentinel_stream = BaseStream()


class SINAR:
    def __init__(self, yolo_model, motas_model, 
                 live_stream: bool = False,
                 *, # keyword-only arguments
                 device: Optional[Union[int, Literal["cpu"]]]=0,
                 probability_threshold: float = 0.5,
                 gang_detected_text: str = "ADA GENG MOTOR"):
        logger.info(f"Initializing SINAR {'live' if live_stream else 'greedy'} mode")
        self.yolo_model = YOLO(yolo_model, task="detect")
        # self.yolo_model.to(0)
        self.device = device
        # self.yolo_model.fuse()
        self.live_stream = live_stream
        
        self._tracks = defaultdict(list)
        
        # self.device = device
        logger.info(f"yolo model loaded [{yolo_model}]")
        self.motas = Motas(motas_model, live_stream=live_stream, threshold=probability_threshold, device=device)
        self.geng_detected = False
        self.detected_action : Optional[Callable] = None
        self.gang_detected_text = gang_detected_text
        self.benchmark_mode = False

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
        logger.trace(f"{result.verbose()} speed: {sum(result.speed.values()):.2f} ms")

        # put result to analysis behavior predictor
        self.motas.put_result(result.cpu())

        # predict on demand if ready and in live stream mode
        if self.live_stream and self.motas.ready():
            self.geng_detected = self.motas.predict()
            if self.geng_detected and self.detected_action is not None: # do only once
                self.detected_action()

        if not self.benchmark_mode:
            frame = self._annotate(result)
            # do as long as pred is true
            if self.geng_detected:
                frame = cvtext(frame, self.gang_detected_text)

        return frame
    
    def __call__(self, source,
                 streamto: BaseStream = _sentinel_stream, 
                 frame_preprocessor : Optional[Callable] = None, 
                 stop_event: Optional[Event] = None):
        self.main_loop(source, streamto, frame_preprocessor, stop_event)
    
    def main_loop(self, source,
                  *,
                 streamto: BaseStream = _sentinel_stream, 
                 frame_postprocessor : Optional[Callable] = None, 
                 stop_event: Optional[Event] = None,
                 benchmark_mode: bool = False) -> Optional[MotasResult]:
        """
        Sinar main loop

        Args:
            source (str): source of the stream, could be a file path or an url
            streamto (BaseStream, optional): stream to write the result. Defaults to _sentinel_stream.
            frame_preprocessor (Optional[Callable], optional): frame preprocessor. Defaults to None.
            stop_event (Optional[Event], optional): stop event. Defaults to None.

        """
        if streamto is not _sentinel_stream and benchmark_mode:
            raise ValueError("benchmark_mode cannot be used while streaming, remove streamto param")
        self.benchmark_mode = benchmark_mode

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
        while (running := capture.isOpened()):
            ret, frame = capture.read()
            if not ret:
                break

            frame = self.predict_frame(frame)

            if not self.benchmark_mode:
                if frame_postprocessor is not None and callable(frame_postprocessor):
                    frame = frame_postprocessor(frame)

                # write to stream
                running = streamto.write(frame)
            
            # stop event
            if (stop_event is not None and stop_event.is_set()) or not running:
                break
        # clear tracks
        self._tracks.clear()
        logger.info("tracker stop")
        # stop analysis behavior predictor
        # self.ab_predictor.stop()
        streamto.stop()
        logger.info("stream stopped")
        capture.release()
        if not self.live_stream:
            motas_res = self.motas.predict()
            logger.info(f"Motion Analysis result: {'GENG MOTOR' if motas_res.is_gang else 'aman 👌'}")
            return motas_res
    
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
                self._draw_centroid_and_tracks(annotator, tracks, color=colors(int(track_id)))
        return annotator.result()

    def _draw_centroid_and_tracks(self, annotator: Annotator, tracks, color, centroid_radius=3, trail_thickness=1):
        if not tracks or len(tracks) < 1:
            return

        # Access image directly from Annotator (the internal image reference)
        im = annotator.im if hasattr(annotator, "im") else annotator.result()

        # Draw line trails between consecutive centroids
        for (x0, y0), (x1, y1) in zip(tracks[:-1], tracks[1:]):
            cv2.line(im, (int(x0), int(y0)), (int(x1), int(y1)), color, trail_thickness)

        # Draw current centroid
        x, y = tracks[-1]
        cv2.circle(im, (int(x), int(y)), centroid_radius, color, -1)  
        # return im  

    
    def reset(self):
        self._tracks.clear()
        self.motas.reset()
        self.geng_detected = False
        self.yolo_model.predictor.trackers[0].reset() # Reset the tracker to make sure it doesn't hold state from previous video
        logger.info("SINAR state reset ✔")

    
