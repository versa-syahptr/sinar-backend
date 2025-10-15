import subprocess
import numpy as np
import os
import cv2

from sinar.utils import VideoInfo
from sinar.logger import get

logger = get(__name__)


RTMP_URL="rtmp://a.rtmp.youtube.com/live2/"
YTSTREAM = RTMP_URL + os.environ.get("SINAR_YT_KEY", '')

class BaseStream:
    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.fps = 0
    
    def set_video_info(self, w, h, fr):
        self.width = w
        self.height = h
        self.fps = fr
    
    def validate_video_info(self) -> bool:
        if self.width == 0 or self.height == 0 or self.fps == 0:
            raise ValueError("Video info not set, call set_video_info first")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def start(self, output : str):
        pass
    def write(self, frame: np.ndarray) -> bool:
        return True # make sure the stream is still running, even if not implemented
    def stop(self):
        pass

class RTMPStream(BaseStream):
    def __init__(self, w, h, fr):
        self.cmd = [
            'ffmpeg',
            '-re',
            '-y',  # Overwrite output file if it already exists
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{w}x{h}",  # Use the same resolution as the input video
            '-r', str(fr),  # Use the same frame rate as the input video
            '-i', '-',  # Input from stdin
            '-f', 'lavfi',
            '-i', 'anullsrc', # null audio
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-maxrate', '8m',
            '-bufsize', '10m',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-g', '50',
            '-f', 'flv',
            # Omit output
        ]
        self.proc = None

    def start(self, output):
        self.cmd.append(output)
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
        return self
    
    def write(self, frame: np.ndarray):
        if self.proc is None:
            raise Exception("Stream not started")
        self.proc.stdin.write(frame.tobytes())

    def stop(self):
        if self.proc is None:
            return
        self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait()
        self.proc = None


class Viewer(BaseStream):
    def __init__(self, title="result", stop_key='q'):
        """
        Viewer class to display the result

        Args: 
            title (str, optional): title of the window. Defaults to "result".
            stop_key (str, optional): key to stop the viewer. Defaults to 'q'.
        
        """
        self.title = title
        self.stop_key = stop_key

    def write(self, frame: np.ndarray, delay_ms=10) -> bool:
        """
        Write frame to the viewer
        
        Args: 
            frame (np.ndarray): frame to write
            
        Returns: 
            bool: True if the viewer is still running, False otherwise
        """
        title = f"{self.title} [press '{self.stop_key}' to stop]"
        cv2.imshow(title, frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord(self.stop_key):
            return False
        return True
    
    def stop(self):
        return cv2.destroyAllWindows()
    

class Saver(BaseStream): # attach directly to BaseStream instead of Viewer to avoid opening a window
    def __init__(self, video_info: VideoInfo, output="output.mp4"):
        """
        Saver class to save the result to a file

        Args:
            w (int): width of the video
            h (int): height of the video
            fps (int): frame per second
            title (str, optional): title of the window. Defaults to "result".
            output (str, optional): output file. Defaults to "output.mp4".
        
        """
        super().__init__()
        self.output = output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output, fourcc, video_info.fps, (video_info.width, video_info.height))

    def write(self, frame: np.ndarray):
        self.writer.write(frame)
        return super().write(frame)
    
    def stop(self):
        self.writer.release()
        return super().stop()
    
    def __repr__(self):
        return f"{self.__class__.__name__} to {self.output}"


class MultiStream(BaseStream):
    def __init__(self, streams: list[BaseStream]):
        logger.info(f"Using MultiStream for {[x for x in streams]}")
        self.streams = streams
        super().__init__()
    
    def set_video_info(self, w, h, fr):
        for s in self.streams:
            s.set_video_info(w, h, fr)
        return super().set_video_info(w, h, fr)
    
    def validate_video_info(self) -> bool:
        return super().validate_video_info()
    
    def write(self, frame: np.ndarray):
        for s in self.streams:
            if not s.write(frame):
                return False
        return True
    
    def start(self, output: str):
        for s in self.streams:
            s.start(output)
        return self
    
    def stop(self):
        for s in self.streams:
            s.stop()
        return super().stop()
    