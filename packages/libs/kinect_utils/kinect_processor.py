from datetime import time
import json
import freenect
from nats.js import JetStreamManager
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from freenect import DEPTH_MM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from libs.kinect_utils.frame_convert import video_cv
import time

console = Console()

# Type aliases for clarity
ImageFrame = np.ndarray
DepthMap = np.ndarray


class CameraCalibrator:
    def __init__(self, calibration_file: str) -> None:
        self.load_calibration(calibration_file)

    def load_calibration(self, calibration_file: str) -> None:
        with np.load(calibration_file) as X:
            self.mtx, self.dist = X["mtx"], X["dist"]
        console.print(Panel("[bold green]Calibration loaded successfully[/bold green]"))
        time.sleep(2)

    def undistort_frame(self, frame: ImageFrame) -> ImageFrame:
        h, w = frame.shape[:2]
        newCameraMtx, roi = cv.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h)
        )

        dst = cv.undistort(frame, self.mtx, self.dist, None, newCameraMtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]

        return dst


class ImageProcessor:
    def __init__(self, sharpen_method: str = "kernel", denoise_method: str = "median"):
        self.sharpen_method = sharpen_method
        self.denoise_method = denoise_method

    def denoise_frame(self, frame) -> ImageFrame:
        if self.denoise_method == "bilateral":
            return cv.bilateralFilter(frame, d=1, sigmaColor=75, sigmaSpace=75)
        elif self.denoise_method == "nlmeans":
            return cv.fastNlMeansDenoisingColored(
                frame, h=10, hColor=10, templateWindowSize=75
            )
        elif self.denoise_method == "median":
            return cv.medianBlur(frame, 3)
        elif self.denoise_method == "none":
            return frame
        else:
            raise ValueError(
                "Invalid denoise_method. Use 'bilateral', 'nlmeans', or 'non'"
            )

    def erode_mask(self, mask, kernel_size=(3, 3), iterations=1):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
        eroded_mask = cv.erode(mask, kernel, iterations=iterations)
        return eroded_mask

    def sharpen_frame(self, frame):
        """
        Appling a sharpening filter to the input frame.
        Args:
            frame: Input image (BGR format).
        Returns:
            Sharpened image
        """
        if self.sharpen_method == "unsharp":
            # Unsharp Mask: Blur the image and subtract to en
            blurred = cv.GaussianBlur(frame, (5, 5), 0)
            return cv.addWeighted(frame, 1.5, blurred, -0.5, 0)
        elif self.sharpen_method == "kernel":
            # Kernel-based sharpening
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            return cv.filter2D(frame, -1, kernel)
        else:
            raise ValueError("Invalid sharpen_method.  Use 'unsharp' or 'kernel'")

    def contrast_image(self, frame):
        alpha = 1.2
        beta = 0
        return cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def intensify_image(self, frame):
        # Convert to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # h, s, v = cv.split(hsv)

        # Blue hue is around 120
        lower_blue = np.array([110, 80, 80])
        upper_blue = np.array([140, 255, 255])

        # Create mask for blue regions
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        # Apply errosion to clean up the mask
        # mask = self.erode_mask(mask, kernel_size=(3, 3), iterations=1)

        kernel = np.ones((3, 3), np.uint8)  # Small kernel for subtle noise removal

        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

        hsv[:, :, 1] = np.where(mask, hsv[:, :, 1] * 1.3, hsv[:, :, 1])
        hsv[:, :, 2] = np.where(mask, hsv[:, :, 2] * 1.1, hsv[:, :, 2])

        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    def enhance_hdr(self, frame):
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        le, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
        le = clahe.apply(le)
        lab = cv.merge((le, a, b))
        return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


class DepthProcessor:
    @staticmethod
    def get_depth() -> DepthMap:
        depth, _ = freenect.sync_get_depth(format=DEPTH_MM)
        return depth

    @staticmethod
    def get_video() -> ImageFrame:
        return video_cv(freenect.sync_get_video()[0])

    @staticmethod
    def stop_video() -> None:
        freenect.sync_stop()

    @classmethod
    def get_center_depth(cls, center_x, center_y) -> float:
        depth_map = cls.get_depth()
        height, width = depth_map.shape

        if 0 <= center_x < width and 0 <= center_y < height:
            return float(depth_map[center_y, center_x])
        else:
            return 0.0

    def get_real_world_coordinates(
        self, pixel_x: int, pixel_y: int, depth: float, calibrator: CameraCalibrator
    ) -> tuple:
        # Convert pixel coordinates to normalized device coordinates
        ndc_x = (pixel_x - calibrator.mtx[0, 2]) / calibrator.mtx[0, 0]
        ndc_y = (pixel_y - calibrator.mtx[1, 2]) / calibrator.mtx[1, 1]

        real_x = ndc_x * depth
        real_y = ndc_y * depth
        real_z = depth

        return real_x, real_y, real_z


class KinectProcessor:
    def __init__(
        self,
        detection_model: str,
        obb_model: str,
        calibrator: CameraCalibrator,
        sharpen_method: str = "kernel",
        denoise_method: str = "median",
    ) -> None:
        self.detection_model = YOLO(detection_model)
        self.obb_model: YOLO = YOLO(obb_model)
        self.detection_labels = self.detection_model.names
        self.obb_labels = self.obb_model.names
        self.calibrator = calibrator
        self.image_processor = ImageProcessor(sharpen_method, denoise_method)
        self.depth_processor = DepthProcessor()

    def process_frame(self, frame):
        # Undistort frame
        undistorted_frame = self.calibrator.undistort_frame(frame)

        # Enhance image
        enhance_hdr = self.image_processor.enhance_hdr(undistorted_frame)

        return enhance_hdr

    def detect_objects(self, frame):

        obb_results = self.obb_model(frame)
        obb_frame = obb_results[0].plot(font_size=1)

        # # Publish data to NATS
        # payload = {"x": float(real_x), "y": float(real_y), "z": float(real_z)}
        #
        # await self.js.publish("camera.collected", json.dumps(payload).encode())

        return obb_results, obb_frame

    def calculate_real_world_coordinates(self, obb_results):
        table = Table(title="Object Detection Results")
        table.add_column("Object", style="cyan")
        table.add_column("Center", style="magenta")
        table.add_column("Distance", style="green")
        table.add_column("Real-world Coordinates", style="yellow")

        object_coord_info = {}

        for result in obb_results[0].obb:

            class_index = result.cls[0].item()
            class_name = self.obb_labels[class_index]
            x1, y1, x2, y2, rotation = result.xywhr[0]

            # Calculate center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Get distance based on center points
            distance = self.depth_processor.get_center_depth(center_x, center_y)

            # Draw center point on the frame
            # cv.circle(obb_frame, (center_x, center_y), 20, (0, 255, 0), -1)

            real_x, real_y, real_z = self.depth_processor.get_real_world_coordinates(
                center_x, center_y, distance, self.calibrator
            )

            coord_data = [float(real_x), float(real_y), float(real_z)]

            object_coord_info[class_name] = np.array(coord_data)

            encoded_data = json.dumps(coord_data).encode()

            # await self.js.publish("camera.collected", encoded_data)
            # await send_coordinates(self, coord_data)

            table.add_row(
                f"{class_name.capitalize()}",
                f"({center_x}, {center_y})",
                f"{distance}mm",
                f"X: {real_x:.2f}mm, Y: {real_y:.2f}mm, Z: {real_z:.2f}mm",
            )

        console.print(table)
        return object_coord_info
