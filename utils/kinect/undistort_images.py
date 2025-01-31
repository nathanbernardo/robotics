import cv2 as cv
import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
# # print("ret: ", ret)
# print("camera matrix: ", mtx)
# print("distortion coefficients: ", dist)
# print("rvec: ", rvecs)
# print("tvec: ", tvecs)

console = Console()

def load_calibration_data(file_path):
    data = np.load(file_path)
    return data['ret'], data['mtx'], data['dist'], data['rvecs'], data['tvecs']

def process_video(input_path, output_path, mtx, dist):
    input_video = cv.VideoCapture(input_path)

    fps = input_video.get(cv.CAP_PROP_FPS)
    width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing video...", total=total_frames)

        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break

            undistorted = cv.undistort(frame, mtx, dist, None, newCameraMtx)

            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]

            if undistorted.shape[0] < height or undistorted.shape[1] < width:
                undistorted = cv.resize(undistorted, (width, height))

            out.write(undistorted)
            progress.update(task, advance=1)

    input_video.release()
    out.release()
    cv.destroyAllWindows()

def main():
    console.print(Panel("Video Undistortion Process", style="bold magenta"))

    calibration_file = 'camera_calibration.npz'
    input_video_path = './output.mp4'
    output_video_path = 'output_undistorted.mp4'

    console.print("[yellow]Loading calibration data...")
    _, mtx, dist, _, _ = load_calibration_data(calibration_file)

    console.print("[green]Calibration data loaded successfully")
    console.print(f"[cyan]Processing video: {input_video_path}")

    process_video(input_video_path, output_video_path, mtx, dist)

    console.print(f"[green]Video processing complete. Output saved to: {output_video_path}")

if __name__ == '__main__':
    main()
