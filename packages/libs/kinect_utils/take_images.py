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

    while captured < num_images:
        frame = processor.get_video()
        if frame is None:
            break

        rgb_image = np.copy(frame)
        image_list.append(rgb_image)
        captured += 1

    for i, img in enumerate(image_list):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kinect_image_{timestamp}_{i+1}.png"
        cv.imwrite(filename, img)

    processor.stop_video()
