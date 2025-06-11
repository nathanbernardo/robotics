import asyncio
import time
import cv2 as cv
import json
from config.config import MinioConfig, ModelConfig
from libs.nats.jetstream_manager import JetStreamManager
from libs.kinect_utils.kinect_processor import (
    CameraCalibrator,
    DepthProcessor,
    KinectProcessor,
)
from libs.minio_utils.minio_client import MinioClient
from libs.state_machine.state_machine import State, StateMachine


async def setup_nats():
    jsm = JetStreamManager()
    await jsm.connect()
    await jsm.ensure_stream("robot_events", subjects=["robot.*"], max_msgs=100_100)
    return jsm


async def main():
    jsm = await setup_nats()
    js = jsm.js if jsm.js is not None else None
    print("JS: ", js)
    #
    # # Fetch detection models from Minio
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
    depth_processor = DepthProcessor()
    kinect_processor = KinectProcessor(detection_path, obb_path, calibrator)
    cv.destroyAllWindows()

    try:
        state_machine = StateMachine(js)
        # state_machine.run()

        target_fps = 30
        frame_time = 1 / target_fps
        while True:
            start_time = time.time()
            frame = depth_processor.get_video()
            if frame is None:
                break

            # annotated_frame = await processor.process_frame(frame)
            refined_frame = kinect_processor.process_frame(frame)
            obb_results, annotated_frame = kinect_processor.detect_objects(
                refined_frame
            )

            current_sm_state = state_machine.get_current_state()
            if current_sm_state == State.CALIBRATING:
                print("Calculatin real world coordinates...")
                real_world_coordinates = (
                    kinect_processor.calculate_real_world_coordinates(obb_results)
                )
                print("Asking for coordinate data...")
                state_machine.ask_for_coord_data(real_world_coordinates)
                print("Real world coordinates: ", real_world_coordinates)
            #
            await state_machine.transition()
            #
            #
            # # Publis data to NATS
            # encoded_payload = json.dumps(real_world_coordinates).encode()
            #
            # cv.imshow("YOLOv11 Inference", annotated_frame)
            cv.imshow("YOLOv11 Inference", annotated_frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                print("Stopping sync")
                break
            elapsed = time.time() - start_time
            # await asyncio.sleep(max(0, frame_time - elapsed))
            await asyncio.sleep(1)
            # await asyncio.sleep(2)
    finally:
        print("DONE")
        # processor.stop_video()
        cv.destroyAllWindows()
        # await jsm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
