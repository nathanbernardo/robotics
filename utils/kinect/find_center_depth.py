import freenect
import numpy as np
import cv2
from ultralytics import YOLO
from freenect import DEPTH_MM
from kinect_utils.frame_convert import pretty_depth_cv

def get_depth():
    depth, _ = freenect.sync_get_depth(format=DEPTH_MM)
    return depth

def get_center_depth():
    depth_map = get_depth()
    height, width = depth_map.shape
    center_y, center_x = height // 2, width // 2
    center_depth = depth_map[center_y, center_x]
    return center_depth

def get_video():
    return freenect.sync_get_video()[0]

# Load Pretrain Yolov11 model
model = YOLO("/home/ncbernar/.pyenv/runs/obb/train41/weights/best.pt")


if __name__ == "__main__":
    while True:
        center_depth = get_center_depth()
        print(f"Depth at center: {center_depth} mm")

        frame = get_video()
        # cv2.imshow("Frame: ", frame)
        # results = model(frame)

        # for result in results:
        #     annotated_frame = result.plot()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('FRAME', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    cv2.destroyAllWindows()
