import threading

import cv2, os
from ultralytics import YOLO
from stream import RTMPStream

def run_tracker_in_thread(filename, model):
    video = cv2.VideoCapture(filename)
    # frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    streamer = RTMPStream(w, h, fps)
    streamer.start(f'rtmp://localhost:1935/output/{os.path.basename(filename).split(".")[0]}')
    video.release()
    results_generator = model.track(filename, stream=True)
    for result in results_generator:
    # while video.isOpened():
        # ret, frame = video.read()
        # if ret:
            # results = model.track(source=frame, persist=True)
        res_plotted = result.plot()
        streamer.write(res_plotted)
            # cv2.imshow('p', res_plotted)
            # if cv2.waitKey(1) == ord('q'):
                # break
        # else:
        #     break
    streamer.stop()


# Load the models
# model = YOLO('best.pt')
# model2 = YOLO('best.pt')

# Define the video files for the trackers
srcs = ['https://pelindung.bandung.go.id:3443/video/HIKSVISION/Ant.m3u8',
        'https://pelindung.bandung.go.id:3443/video/HUAWEI/soetabubat.m3u8']
threads = []
for src in srcs:
    model = YOLO('best.pt')
    tracker_thread = threading.Thread(target=run_tracker_in_thread, args=(src, model), daemon=True)
    tracker_thread.start()
    threads.append(tracker_thread)

for thread in threads:
    thread.join()


# Create the tracker threads
# tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1), daemon=True)
# tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2), daemon=True)

# # Start the tracker threads
# tracker_thread1.start()
# tracker_thread2.start()

# # Wait for the tracker threads to finish
# tracker_thread1.join()
# tracker_thread2.join()

# # Clean up and close windows
# cv2.destroyAllWindows()
