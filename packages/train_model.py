from ultralytics import YOLO

from libs.minio_utils.minio_client import MinioClient
from config.config import MinioConfig


def main():
    # Grab model from minio
    minio_client = MinioClient(
        endpoint=MinioConfig.SERVER,
        access_key=MinioConfig.ACCESS_KEY,
        secret_key=MinioConfig.SECRET_KEY,
    )

    obb_path = "./tmp/obb_2025_05_21.pt"
    minio_client.download_file("ai-models", "obb_2025_05_21.pt", obb_path)

    # Train model with new data
    model = YOLO(obb_path)
    model.train(
        data="/home/ncbernar/datasets/2025/core_obb_dataset/data.yaml",
        epochs=500,
        imgsz=640,
    )


if __name__ == "__main__":
    main()
