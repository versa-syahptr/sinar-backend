from ultralytics import YOLO

from stream import RTMPStream, YTSTREAM
from predigenk import Anbev
from utils import cvtext
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
        print(f"yolo model loaded [{yolo_model}]")
        self.ab_predictor = Anbev(abModel)
    
    def __call__(self, source, streamto: RTMPStream = None, frame_preprocessor=None):
        result_generator = self.yolo_model.track(source, device=self.device, stream=True)
        # start thread if it isn't started
        if not self.ab_predictor.is_alive():
            self.ab_predictor.start()
        print(f"tracker start ({source})")
        for result in result_generator:
            frame = result.plot()
            # ANALYSIS BEHAVIOR START
            # logger.debug(result.verbose())
            if result.boxes.id is not None: # ada deteksi
                self.ab_predictor.result_queue.put(result)
                # if (idx_frame == STEP or idx_frame == next_frame_idx) and len(centroids) < MAXSHAPE:
                #     ids, xy = get_xyid(result.boxes)
                #     centroids.append(to_dict(ids, xy, flatten=True))
                #     next_frame_idx = idx_frame + SAMPLING
                #     print(f"{idx_frame} ✔\r")
                # idx_frame += 1
                # if len(centroids) >= MAXSHAPE: # full
                #     df = make_dataframe(centroids)
                #     idx_frame = STEP
                #     next_frame_idx = idx_frame + SAMPLING
                #     # PREDICT
                #     x = fill_square(df.values, MAXSHAPE)
                #     print("anbev predicting ")
                #     preds = self.ab_predict(x)
                #     print("preds result:", "GENG MOTOR" if preds else "aman 👌")
                #     flag = bool(preds)
            # ANALYSIS BEHAVIOR END
            if frame_preprocessor is not None:
                frame = frame_preprocessor(frame)
            # geng motor terdeteksi
            if self.ab_predictor.genk_event.is_set():
                frame = cvtext(frame, "ADA GENG MOTOR")
            # write to stream
            if streamto is not None:
                streamto.write(frame)
