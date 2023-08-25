import argparse
from supervision.video.dataclasses import VideoInfo
import time
import ultralytics
import sinar 
from stream import RTMPStream, YTSTREAM
import multiprocessing as mp
import logger
import os

logger = logger.get(__name__)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yolo", type=str, required=True, help="path to yolo model")
    parser.add_argument("-a", "--ab", type=str, required=True, help="path to analysis behavior model")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="device to run yolo model")
    parser.add_argument("-s", "--source", type=str, required=True, action="append", help="source(s) of video")
    parser.add_argument("-o", "--output", type=str, required=True, action="append", help="output stream(s)")

    args = parser.parse_args()

    if len(args.source) != len(args.output):
        parser.error("Different lenght of source and output stream!")

    ultralytics.checks()

    # sinar = SINAR(args.yolo, args.ab)

    try:
        for source, output in zip(args.source, args.output):
            logger.info(f"processing {source} -> {output}")
            # vi = VideoInfo.from_video_path(source)
            # streamer = RTMPStream(vi.width, vi.height, vi.fps).start(output)
            # sinar.start_threaded(os.path.basename(source), source, streamto=streamer)
            sinar.new_sinar_process(args.yolo, args.ab, source, output, 
                                    process_name=os.path.basename(source),device=args.device)
        logger.info("all process started, main process idling")
        while True:
            pass
    except KeyboardInterrupt:
        # sinar.stop_all()
        sinar.stop_all_processes() 
