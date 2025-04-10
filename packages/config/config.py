import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEBUG = False


class MinioConfig(Config):
    SERVER = os.environ.get("MINIO_SERVER", "")
    ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
    SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
