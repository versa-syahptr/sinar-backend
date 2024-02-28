import subprocess
import numpy as np
import os

RTMP_URL="rtmp://a.rtmp.youtube.com/live2/"
YTSTREAM = RTMP_URL + os.environ.get("SINAR_YT_KEY", '')

class BaseStream:
    def __init__(self) -> None:
        pass
    def start(self, output : str):
        pass
    def write(self, frame: np.ndarray):
        pass
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