import asyncio
import cv2 as cv
from libs.nats.jetstream_manager import JetStreamManager
from libs.kinect_utils.kinect_processor import KinectProcessor
from libs.minio_utils.minio_client import MinioClient

# Define constants
MODEL_PATHS = {
    "detection": "/home/ncbernar/Downloads/detect_9.pt",
    "obb": "/home/ncbernar/Downloads/obb_58.pt",
}
CALIBRATION_FILE = (
    "/home/ncbernar/Code/nats_sandbox/packages/calibration/camera_calibration.npz"
)


async def main():
    # Fetch detection models from Minio
    minio_client = MinioClient(
        endpoint="localhost:9000", access_key="dev", secret_key="dev-minio"
    )

    model_path = "./tmp/detect_9.pt"
    detection_model = minio_client.download_file("ai-models", "detect_9.pt", model_path)
    print(detection_model)

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
    processor = KinectProcessor(model_path, MODEL_PATHS["obb"], CALIBRATION_FILE, js)

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
