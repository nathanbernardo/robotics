import asyncio
import cv2 as cv
from config.config import MinioConfig
from libs.nats.jetstream_manager import JetStreamManager
from libs.kinect_utils.kinect_processor import KinectProcessor
from libs.minio_utils.minio_client import MinioClient

# Define constants
CALIBRATION_FILE = (
    "/home/ncbernar/Code/nats_sandbox/packages/calibration/camera_calibration.npz"
)


async def main():
    # Fetch detection models from Minio
    minio_client = MinioClient(
        endpoint=MinioConfig.SERVER,
        access_key=MinioConfig.ACCESS_KEY,
        secret_key=MinioConfig.SECRET_KEY,
    )

    detection_path = "./tmp/detect_coco128_200epochs.pt"
    obb_path = "./tmp/obb_58.pt"
    minio_client.download_file(
        "ai-models", "detect_coco128_200epochs.pt", detection_path
    )
    minio_client.download_file("ai-models", "obb_58.pt", obb_path)

    # Connect to Nats JetStream
    jsm = JetStreamManager()
    await jsm.connect()
    js = jsm.nc.jetstream()

    # Ensure stream exists
    await jsm.ensure_stream(
        "camera_events",
        subjects=["camera.*"],
        max_msgs=100_000,
    )

    # Instantiate the Kinect Processor
    processor = KinectProcessor(detection_path, obb_path, CALIBRATION_FILE, js)

    while True:
        frame = processor.get_video()
        annotated_frame = await processor.process_frame(frame)

        cv.imshow("YOLOv11 Inference", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()
    await jsm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
