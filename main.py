import argparse
from supervision.video.dataclasses import VideoInfo
import cv2
import ultralytics
from sinar import SINAR
from stream import RTMPStream, YTSTREAM
import logger
import os

logger = logger.get(__name__)

if __name__ == "__main__":
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

    sinar = SINAR(args.yolo, args.ab)

    try:
        for source, output in zip(args.source, args.output):
            logger.info(f"processing {source} -> {output}")
            vi = VideoInfo.from_video_path(source)
            streamer = RTMPStream(vi.width, vi.height, vi.fps).start(output)

            sinar.start_threaded(os.path.basename(source), source, streamto=streamer)
        logger.info("all process started, main process idling")
        while True:
            pass
    except KeyboardInterrupt:
        sinar.stop_all()
