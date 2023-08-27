import argparse
from supervision.video.dataclasses import VideoInfo
import time
import ultralytics
# from sinar import SINAR
import sinar
from stream import RTMPStream, YTSTREAM
# import multiprocessing as mp
import logger
import os

logger = logger.get(__name__)

if __name__ == "__main__":
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yolo", type=str, required=True, help="path to yolo model")
    parser.add_argument("-a", "--ab", type=str, required=True, help="path to analysis behavior model")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device to run yolo model")
    parser.add_argument("-s", "--source", type=str, required=True, help="source(s) of video", action="append")
    parser.add_argument("-o", "--output", type=str, required=True, help="output stream(s)", action="append")

    args = parser.parse_args()

    if len(args.source) != len(args.output):
        parser.error("Different lenght of source and output stream!")

    ultralytics.checks()

    # sinar = SINAR(args.yolo, args.ab)

    try:
        # single source
        # logger.info(f"processing {args.source} -> {args.output}")
        # vi = VideoInfo.from_video_path(args.source)
        # streamer = RTMPStream(vi.width, vi.height, vi.fps).start(args.output)
        # sinar(args.source, streamto=streamer)
        for source, output in zip(args.source, args.output):
            logger.info(f"processing {source} -> {output}")
            # vi = VideoInfo.from_video_path(source)
            # streamer = RTMPStream(vi.width, vi.height, vi.fps).start(output)
            # sinar.start_threaded(os.path.basename(source), source, streamto=streamer)
            sinar.new_sinar_process(args.yolo, args.ab, source, output, 
                                    process_name=os.path.basename(source))
        logger.info("all process started, main process idling")
        while True:
            time.sleep(10)
        # for p in sinar._process:
        #     p.process.join()
            # logger.info(f"{p.name} stopped")

    except Exception:
        logger.exception("error in main process")
        # sinar.stop_all()
        # sinar.stop_all_processes() 
        # sinar.ab_predictor.stop()
        # streamer.stop()
    except KeyboardInterrupt:
        pass
    finally:
        # sinar.stop_all()
        sinar.stop_all_processes() 
        # sinar.ab_predictor.stop()
        # streamer.stop()
        logger.info("main process stopped")

