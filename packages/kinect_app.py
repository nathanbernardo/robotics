import asyncio
import time
import cv2 as cv
from config.config import MinioConfig, ModelConfig
from libs.nats.jetstream_manager import JetStreamManager
from libs.kinect_utils.kinect_processor import CameraCalibrator, KinectProcessor
from libs.minio_utils.minio_client import MinioClient


async def setup_nats():
    jsm = JetStreamManager()
    await jsm.connect()
    await jsm.ensure_stream("camera_events", subjects=["camera.*"], max_msgs=100_100)
    return jsm


async def main():
    # jsm = await setup_nats()

    # Fetch detection models from Minio
    minio_client = MinioClient(
        endpoint=MinioConfig.SERVER,
        access_key=MinioConfig.ACCESS_KEY,
        secret_key=MinioConfig.SECRET_KEY,
    )

    detection_path = ModelConfig.DETECTION_MODEL_PATH
    obb_model = ModelConfig.OBB_MODEL
    obb_path = ModelConfig.OBB_MODEL_PATH

    minio_client.download_file("ai-models", obb_model, obb_path)

    # Instantiate the Kinect Processor
    calibrator = CameraCalibrator(ModelConfig.CALIBRATION_FILE)
    processor = KinectProcessor(detection_path, obb_path, calibrator)
    cv.destroyAllWindows()

    try:
        target_fps = 30
        frame_time = 1 / target_fps
        while True:
            start_time = time.time()
            frame = processor.get_video()
            if frame is None:
                break

            # annotated_frame = await processor.process_frame(frame)
            refined_frame = await processor.process_frame(frame)
            annotated_frame = processor.detect_objects(refined_frame)

            cv.imshow("YOLOv11 Inference", annotated_frame)
            # if cv.getWindowProperty("YOLOv11 Inference", cv.WND_PROP_VISIBLE) < 1:
            #     break

            if cv.waitKey(1) & 0xFF == ord("q"):
                print("Stopping sync")
                break
            elapsed = time.time() - start_time
            await asyncio.sleep(max(0, frame_time - elapsed))
    finally:
        processor.stop_video()
        cv.destroyAllWindows()
        # await jsm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
