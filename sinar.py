from ultralytics import YOLO
# from threading import Thread, Event
from multiprocessing import Process, Event
import time
from supervision.video.dataclasses import VideoInfo

from stream import RTMPStream, YTSTREAM
from predigenk import Anbev
from utils import cvtext, Process_wrapper, check_stream
import logger
import cv2

MAXSHAPE = 30
SAMPLING = 5
STEP = 1

logger = logger.get(__name__)

sinar_processes = []

class SINAR:
    def __init__(self, yolo_model, abModel):
        self.yolo_model = YOLO(yolo_model)
        # self.yolo_model.to(0)
        self.yolo_model.fuse()
        
        # self.device = device
        logger.info(f"yolo model loaded [{yolo_model}]")
        self.ab_predictor = Anbev(abModel, threaded=True)
        
        # processes
        self._process = []
    
    def __call__(self, source,
                 streamto: RTMPStream = None, 
                 frame_preprocessor=None, 
                 stop_event: Event = None):

        # check stream availability
        retry_count = 0
        while not check_stream(source):
            logger.info(f"stream {source} is offline, retrying...")
            time.sleep(5)

        #open source stream
        # cap = cv2.VideoCapture(source)
        # if not cap.isOpened():
        #     raise Exception(f"cannot open stream {source}")
        

        result_generator = self.yolo_model.track(source, device=0, stream=True, verbose=False)
        # start thread if it isn't started
        # if not self.ab_predictor.is_alive():
            # self.ab_predictor.start()
        logger.info(f"tracker start ({source})")
        for result in result_generator:
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
            
            # results = self.yolo_model.track(frame, device=0, verbose=False, persist=True)
            # result = results[0]

            frame = result.plot()
            logger.debug(f"{result.verbose()}; speed: {sum(result.speed.values()):.2f}ms")
            # send result to analysis behavior predictor
            # self.ab_predictor.result_queue.put(result.cpu())

            # put result to analysis behavior predictor
            self.ab_predictor.put_result(result.cpu())
            # predict
            if self.ab_predictor.ready():
                pred = self.ab_predictor.predict()
                if pred:
                    frame = cvtext(frame, "ADA GENG MOTOR")
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
        # self.ab_predictor.stop()
        streamto.stop()
        logger.info("stream stopped")
    
    def start_threaded(self, name, source, streamto: RTMPStream = None, frame_preprocessor=None):
        stop_event = Event()
        p = Thread(target=self, args=(source, streamto, frame_preprocessor, stop_event), name=name)
        p.start()
        self._process.append(Process_wrapper(p, name, stop_event))
        return p
    
    def stop_all(self):
        for p in self._process:
            p.stop_event.set()
            p.process.join()
            logger.info(f"{p.name} stopped")
        logger.info("all process stopped")

def _inner_main(yolo, anbev, source, output, stop_event=None):
    sinar = SINAR(yolo, anbev)
    vi = VideoInfo.from_video_path(source)
    streamer = RTMPStream(vi.width, vi.height, vi.fps).start(output)
    try:
        sinar(source, streamto=streamer, stop_event=stop_event)
    except Exception as e:
        logger.exception(f"error in process: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        # sinar.ab_predictor.stop()
        streamer.stop()

def new_sinar_process(yolo_model, abModel, source, output ,*, process_name=None):
    # sinar = SINAR(yolo_model, abModel)
    stop_event = Event()
    process = Process(target=_inner_main, name=process_name,
                      args=(yolo_model, abModel, source, output, stop_event))
    process.start()
    sinar_processes.append(Process_wrapper(process, process_name, stop_event))

    return process.is_alive()

def stop_process(process_name):
    for p in sinar_processes:
        if p.name == process_name:
            p.stop_event.set()
            p.process.join()
            logger.info(f"{p.name} stopped")
            return True
    return False

def stop_all_processes():
    for p in sinar_processes:
        p.stop_event.set()
        p.process.join()
        # p.process.terminate()
        logger.info(f"{p.name} stopped")
    logger.info("all process stopped")

