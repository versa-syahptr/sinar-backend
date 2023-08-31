import argparse
import time
import ultralytics
import sinar
import multiprocessing as mp
import os
from sinar.logger import logger


if __name__ == "__main__":
    mp.set_start_method("spawn")
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

    try:
        for source, output in zip(args.source, args.output):
            logger.info(f"processing {source} -> {output}")
            sinar.new_sinar_process(args.yolo, args.ab, source, output, 
                                    process_name=os.path.basename(source))
            
        logger.info("all process started, main process idling")
        while sinar.any_process_alive():
            time.sleep(1)

    except Exception:
        logger.exception("error in main process")
    except KeyboardInterrupt:
        pass
    finally:
        sinar.stop_all_processes() 
        logger.info("main process stopped")

