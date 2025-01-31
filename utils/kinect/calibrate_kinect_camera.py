import numpy as np
import cv2 as cv
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

console = Console()

def initialize_chessboard():
    critera = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7,0:6].T.reshape(-1, 2)

    return critera, objp


def process_video(input_video_path):
    criteria, objp = initialize_chessboard()
    input_video = cv.VideoCapture(input_video_path)
    objpoints, imgpoints = [], []
    h, w = 0, 0

    total_frames = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))

    with Progress() as progress:
        task = progress.add_task("[green]Processing video...", total=total_frames)
        
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break;

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]

            ret2, corners = cv.findChessboardCorners(gray, (7, 6), None)

            if ret2:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

            progress.update(task, advance=1)

        input_video.release()
        return objpoints, imgpoints, (w, h)

def calibrate_camera(objpoints, imgpoints, image_size):
    return cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

def save_calibration(calibration_data, filename):
    np.savez(filename, **calibration_data)

def main():
    input_video_path = './output.mp4'
    calibration_file = 'camera_calibration.npz'

    console.print(Panel("Starting camera calibration process", style="bold blue"))

    objpoints, imgpoints, image_size = process_video(input_video_path)
    console.print(f"[yellow]Image size: {image_size}")

    console.print("[cyan]Calibration camera...")
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size)

    calibration_data = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }

    save_calibration(calibration_data, calibration_file)
    console.print(f"[green]Saved camera calibration values to {calibration_file}")

    mean_error = 0 
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    console.print("[yellow]Total error: {}".format(mean_error/len(objpoints)))

if __name__ == '__main__':
    main()

