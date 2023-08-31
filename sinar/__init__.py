from multiprocessing import Event, Process
from supervision.video.dataclasses import VideoInfo

from .sinar import SINAR
from .stream import RTMPStream
from .utils import Process_wrapper
from .logger import logger

# logger = log_module.get(__name__)

sinar_processes = {}

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
        streamer.stop()

def new_sinar_process(yolo_model, abModel, source, output ,*, process_name):
    if process_name in sinar_processes:
        return False
    stop_event = Event()
    process = Process(target=_inner_main, name=process_name,
                      args=(yolo_model, abModel, source, output, stop_event))
    process.start()
    sinar_processes[process_name] = Process_wrapper(process, stop_event)

    return process.is_alive()

def stop_process(process_name):
    if process_name not in sinar_processes:
        return False
    
    sinar_processes[process_name].stop_event.set()
    sinar_processes[process_name].process.join()
    del sinar_processes[process_name]
    logger.info(f"{process_name} stopped")
    return True

def stop_all_processes():
    for pname in list(sinar_processes.keys()):
        stop_process(pname)
    logger.info("all process stopped")

def any_process_alive():
    return any(p.is_alive() for p,_ in sinar_processes.values())