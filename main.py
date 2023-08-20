import argparse
from supervision.video.dataclasses import VideoInfo
import cv2
import ultralytics
from sinar import SINAR
from stream import RTMPStream, YTSTREAM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yolo", type=str, required=True, help="path to yolo model")
    parser.add_argument("-a", "--ab", type=str, required=True, help="path to analysis behavior model")
    parser.add_argument("-s", "--source", type=str, required=True, help="source of video")
    parser.add_argument("-o", "--output", type=str, help="output stream")
    parser.add_argument("--youtube", action="store_true", help="stream to youtube")
    parser.add_argument("--save", action="store_true", help="save video")
    args = parser.parse_args()

    ultralytics.checks()

    vi = VideoInfo.from_video_path(args.source)
    if not args.save:
        streamer = RTMPStream(vi.width, vi.height, vi.fps)
        if args.youtube:
            streamer.start(YTSTREAM)
        else:
            streamer.start(args.output)
    else:
        streamer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), vi.fps, vi.resolution)

    sinar = SINAR(args.yolo, args.ab)
    sinar(args.source, streamto=streamer)