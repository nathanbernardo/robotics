import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEBUG = False


class MinioConfig(Config):
    SERVER = os.environ.get("MINIO_SERVER", "")
    ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
    SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")


class ModelConfig(Config):
    CALIBRATION_FILE = (
        "/home/ncbernar/Code/nats_sandbox/packages/calibration/camera_calibration.npz"
    )
    DETECTION_MODEL = "detect_coco128_200epochs.pt"
    DETECTION_MODEL_PATH = f"./tmp/{DETECTION_MODEL}"
    OBB_MODEL = "obb_2025_05_13.pt"
    OBB_MODEL_PATH = f"./tmp/{OBB_MODEL}"
