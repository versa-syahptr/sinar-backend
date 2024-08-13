import subprocess
import numpy as np
import os
import cv2


RTMP_URL="rtmp://a.rtmp.youtube.com/live2/"
YTSTREAM = RTMP_URL + os.environ.get("SINAR_YT_KEY", '')

class BaseStream:
    def __init__(self) -> None:
        pass
    def start(self, output : str):
        pass
    def write(self, frame: np.ndarray) -> bool:
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

    def write(self, frame: np.ndarray) -> bool:
        """
        Write frame to the viewer
        
        Args: 
            frame (np.ndarray): frame to write
            
        Returns: 
            bool: True if the viewer is still running, False otherwise
        """
        cv2.imshow(self.title, frame)
        if cv2.waitKey(1) & 0xFF == ord(self.stop_key):
            return False
        return True
    def stop(self):
        return cv2.destroyAllWindows()
    

class Saver(Viewer):
    def __init__(self,  w, h, fps, title="result", output="output.mp4"):
        """
        Saver class to save the result to a file

        Args:
            w (int): width of the video
            h (int): height of the video
            fps (int): frame per second
            title (str, optional): title of the window. Defaults to "result".
            output (str, optional): output file. Defaults to "output.mp4".
        
        """
        super().__init__(title)
        self.output = output
        self.writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))

    def write(self, frame: np.ndarray):
        self.writer.write(frame)
        return super().write(frame)
    
    def stop(self):
        self.writer.release()
        return super().stop()