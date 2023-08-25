from ultralytics import YOLO
from multiprocessing import Process, Event
import time

from stream import RTMPStream, YTSTREAM
from predigenk import Anbev
from utils import cvtext, Process_wrapper, check_stream
import logger

MAXSHAPE = 30
SAMPLING = 5
STEP = 1

logger = logger.get(__name__)

class SINAR:
    def __init__(self, yolo_model, abModel, *, device="cpu"):
        self.yolo_model = YOLO(yolo_model)
        self.yolo_model.fuse()
        self.device = device
        logger.info(f"yolo model loaded [{yolo_model}]")
        self.ab_predictor = Anbev(abModel)
        
        # processes
        self._process = []
    
    def __call__(self, source,*, 
                 streamto: RTMPStream = None, 
                 frame_preprocessor=None, 
                 stop_event: Event = None):
        # check stream availability
        while not check_stream(source):
            logger.info(f"stream {source} is offline, retrying...")
            time.sleep(5)

        result_generator = self.yolo_model.track(source, device=self.device, stream=True, verbose=False)
        # start thread if it isn't started
        if not self.ab_predictor.is_alive():
            self.ab_predictor.start()
        logger.info(f"tracker start ({source})")
        for result in result_generator:
            frame = result.plot()
            logger.debug(result.verbose())
            # send result to analysis behavior predictor
            self.ab_predictor.result_queue.put(result)

            if frame_preprocessor is not None:
                frame = frame_preprocessor(frame)
            # geng motor terdeteksi
            if self.ab_predictor.genk_event.is_set():
                frame = cvtext(frame, "ADA GENG MOTOR")
            # write to stream
            if streamto is not None:
                streamto.write(frame)
            # stop event
            if stop_event is not None and stop_event.is_set():
                break
        logger.info("tracker stop")
        # stop analysis behavior predictor
        self.ab_predictor.stop()
        streamto.stop()
        logger.info("stream stopped")
    
    def start_threaded(self, name, source, streamto: RTMPStream = None, frame_preprocessor=None):
        stop_event = Event()
        p = Process(target=self, args=(source, streamto, frame_preprocessor, stop_event), name=name)
        p.start()
        self._process.append(Process_wrapper(p, name, stop_event))
        return p
    
    def stop_all(self):
        for p in self._process:
            p.stop_event.set()
            p.process.join()
            logger.info(f"{p.name} stopped")
        logger.info("all process stopped")
