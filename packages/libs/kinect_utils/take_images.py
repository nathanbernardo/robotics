import numpy as np
import cv2 as cv
from config.config import MinioConfig, ModelConfig
from libs.kinect_utils.kinect_processor import CameraCalibrator, KinectProcessor
from libs.minio_utils.minio_client import MinioClient
from datetime import datetime


def capture_kinect_images(num_images=1):
    captured = 0
    image_list = []

    minio_client = MinioClient(
        endpoint=MinioConfig.SERVER,
        access_key=MinioConfig.ACCESS_KEY,
        secret_key=MinioConfig.SECRET_KEY,
    )

    detection_path = ModelConfig.DETECTION_MODEL_PATH
    obb_model = ModelConfig.OBB_MODEL
    obb_path = ModelConfig.OBB_MODEL_PATH

    minio_client.download_file("ai-models", obb_model, obb_path)

    calibrator = CameraCalibrator(ModelConfig.CALIBRATION_FILE)
    processor = KinectProcessor(detection_path, obb_path, calibrator)
    # Blue hue is around 120
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create mask for blue regions

    while captured < num_images:
        frame = processor.get_video()

        if frame is None:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # # Blue hue is around 120
        # lower_blue = np.array([100, 50, 50])
        # upper_blue = np.array([140, 255, 255])

        # Create mask for blue regions
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
        hsv[:, :, 1] = np.where(
            mask, hsv[:, :, 1] * 1.3, hsv[:, :, 1]
        )  # Boost saturation
        hsv[:, :, 2] = np.where(
            mask, hsv[:, :, 2] * 1.1, hsv[:, :, 2]
        )  # Boost brightness

        result = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        image_list.append(result)
        captured += 1

    for i, img in enumerate(image_list):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kinect_image_{timestamp}_{i+1}.png"
        cv.imwrite(filename, img)

    processor.stop_video()
