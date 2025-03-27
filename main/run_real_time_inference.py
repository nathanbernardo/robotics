import time
import freenect
import numpy as np
import cv2 as cv
from ultralytics import YOLO, solutions
from freenect import DEPTH_MM
from kinect_utils.frame_convert import video_cv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
distancecalculator = solutions.DistanceCalculation(
    model="/home/ncbernar/.pyenv/runs/obb/train58/weights/best.pt",  # path to the YOLO11 model file.
    show=True,  # display the output
)

class KinectProcessor:
    def __init__(self, detection_model, obb_model, calibration_file):
        self.detection_model = YOLO(detection_model)
        self.obb_model = YOLO(obb_model)
        self.detection_labels = self.detection_model.names
        self.obb_labels = self.obb_model.names
        self.load_calibration(calibration_file)

    def load_calibration(self, calibration_file):
        with np.load(calibration_file) as X:
            self.mtx, self.dist = X['mtx'], X['dist']
        console.print(Panel("[bold green]Calibration loaded successfully[/bold green]"))

    @staticmethod
    def get_depth():
        depth, _ = freenect.sync_get_depth(format=DEPTH_MM)
        return depth

    @classmethod
    def get_center_depth(cls, center_x, center_y):
        depth_map = cls.get_depth()
        height, width = depth_map.shape
        print(f"[get_center_depth] (height, width): ({height}, {width})")
        if 0 <= center_x < width and 0 <= center_y < height:
            center_depth = depth_map[center_y, center_x]
            return center_depth
        else:
            return None

    @staticmethod
    def get_video():
        return video_cv(freenect.sync_get_video()[0])

    def undistort_frame(self, frame):
        h, w = frame.shape[:2]
        newCameraMtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))


        dst = cv.undistort(frame, self.mtx, self.dist, None, newCameraMtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst

    def get_real_world_coordinates(self, pixel_x, pixel_y, depth):
        # Convert pixel coordinates to normalized device coordinates
        ndc_x = (pixel_x - self.mtx[0, 2]) / self.mtx[0, 0]
        ndc_y = (pixel_y - self.mtx[1, 2]) / self.mtx[1, 1]

        real_x = ndc_x * depth
        real_y = ndc_y * depth
        real_z = depth

        return real_x, real_y, real_z
    
    # def process_detections(self, results, labels, frame, table):
    #     for result in detection_results[0].boxes:
    #         class_index = result.cls[0].item()
    #         class_name = self.detection_labels[class_index]
    #         x1, y1, x2, y2 = result.xyxy[0]
    #
    #         # Calculate center point
    #         center_x = int((x1 + x2) / 2)
    #         center_y = int((y1 + y2) / 2)
    #
    #         # Get distance based on center points
    #         distance = self.get_center_depth(center_x, center_y)
    #
    #         # Draw center point on the frame
    #         cv.circle(detection_frame, (center_x, center_y), 5, (0, 255, 0), -1)
    #
    #         if distance is not None:
    #             real_x, real_y, real_z = self.get_real_world_coordinates(center_x, center_y, distance)
    #             
    #             table.add_row(
    #                 f"{class_name.capitalize()}",
    #                 f"({center_x}, {center_y})",
    #                 f"{distance}mm",
    #                 f"X: {real_x:.2f}mm, Y: {real_y:.2f}mm, Z: {real_z:.2f}mm"
    #             )
    #
    #         else:
    #             table.add_row(
    #                 f"Object {len(table.rows) + 1}",
    #                 f"({center_x}, {center_y})",
    #                 "N/A",
    #                 "N/A"
    #             )

    def process_frame(self, frame):
        undistorted_frame = self.undistort_frame(frame)

        
        # Run both models
        detection_results = self.detection_model(undistorted_frame)
        obb_results = self.obb_model(undistorted_frame)

        # annoted_frame = results[0].plot()

        # Create combined annotations
        obb_frame = obb_results[0].plot()
        detection_frame = detection_results[0].plot()
        # combined_frame = cv.addWeighted(obb_frame, 0.9, detection_frame, 0.9, 0)
        combined_frame = detection_results[0].plot()
        # combined_frame = detection_results[0].plot(img=combined_frame)

        table = Table(title="Object Detection Results")
        table.add_column("Object", style="cyan")
        table.add_column("Center", style="magenta")
        table.add_column("Distance", style="green")
        table.add_column("Real-world Coordinates", style="yellow")

        # print("RESULTS: ", results[0].obb)
        for result in detection_results[0].boxes:
            class_index = result.cls[0].item()
            class_name = self.detection_labels[class_index]
            x1, y1, x2, y2 = result.xyxy[0]

            # Calculate center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Get distance based on center points
            distance = self.get_center_depth(center_x, center_y)

            # Draw center point on the frame
            cv.circle(detection_frame, (center_x, center_y), 20, (0, 255, 0), -1)

            if distance is not None:
                real_x, real_y, real_z = self.get_real_world_coordinates(center_x, center_y, distance)
                
                table.add_row(
                    f"{class_name.capitalize()}",
                    f"({center_x}, {center_y})",
                    f"{distance}mm",
                    f"X: {real_x:.2f}mm, Y: {real_y:.2f}mm, Z: {real_z:.2f}mm"
                )

            else:
                table.add_row(
                    f"Object {len(table.rows) + 1}",
                    f"({center_x}, {center_y})",
                    "N/A",
                    "N/A"
                )

        # for result in obb_results[0].obb:
        #     class_index = result.cls[0].item()
        #     class_name = self.obb_labels[class_index]
        #     x1, y1, x2, y2, _ = result.xywhr[0]
        #
        #     # Calculate center point
        #     center_x = int((x1 + x2) / 2)
        #     center_y = int((y1 + y2) / 2)
        #
        #     # Get distance based on center points
        #     distance = self.get_center_depth(center_x, center_y)
        #

            # Draw center point on the frame
            # cv.circle(combined_frame, (center_x, center_y), 20, (0, 255, 0), -1)

        #     if distance is not None:
        #         real_x, real_y, real_z = self.get_real_world_coordinates(center_x, center_y, distance)
        #         
        #         table.add_row(
        #             f"{class_name.capitalize()}",
        #             f"({center_x}, {center_y})",
        #             f"{distance}mm",
        #             f"X: {real_x:.2f}mm, Y: {real_y:.2f}mm, Z: {real_z:.2f}mm"
        #         )
        #
        #     else:
        #         table.add_row(
        #             f"Object {len(table.rows) + 1}",
        #             f"({center_x}, {center_y})",
        #             "N/A",
        #             "N/A"
        #         )
        console.print(table)
        # time.sleep(1)
        return combined_frame

def main():
    MODEL_PATHS = {
        'detection': "/home/ncbernar/.pyenv/runs/detect/train9/weights/best.pt",
        'obb': "/home/ncbernar/.pyenv/runs/obb/train58/weights/best.pt"
    }
    CALIBRATION_FILE = "../utils/kinect/camera_calibration.npz"
    processor = KinectProcessor(MODEL_PATHS['detection'], MODEL_PATHS['obb'], CALIBRATION_FILE)

    # Initialize distance calculation object

    # with 
    while True:
        frame = processor.get_video()
        annotated_frame = processor.process_frame(frame)

        cv.imshow("YOLOv11 Inference", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break;

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
