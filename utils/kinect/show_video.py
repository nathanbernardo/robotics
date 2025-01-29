import numpy as np
import cv2
from ultralytics import YOLO
import random
from ultralytics.utils.plotting import Annotator, colors
import glob

video = glob.glob('./output_undistored.mp4')

print(len(video))


input_video = cv2.VideoCapture('/home/ncbernar/Github/robotics/utils/kinect/output_undistorted.mp4')
input_video_2 = cv2.VideoCapture('/home/ncbernar/Github/robotics/utils/kinect/output.mp4')
originalFPS = input_video.get(cv2.CAP_PROP_FPS)
print(input_video.isOpened())
frame_delay = int(1000 / 10)
while True:

    ret1, frame1 = input_video.read()
    ret2, frame2 = input_video_2.read()
    if not ret1 or not ret2:
        break

    combined_frame = np.hstack((frame1, frame2))

    cv2.imshow("Circle annotation", combined_frame)

    # Optional: Display the frame (comment out if not needed)
    # cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
input_video.release()
cv2.destroyAllWindows()

