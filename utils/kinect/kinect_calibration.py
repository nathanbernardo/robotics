import numpy as np
import cv2 as cv
import glob

critera = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7,0:6].T.reshape(-1, 2)

input_video = cv.VideoCapture('./output.mp4')

objpoints = []
imgpoints = []

h, w = 0, 0

try:
    while input_video.isOpened():
        ret, frame = input_video.read()

        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        ret2, corners =  cv.findChessboardCorners(gray, (7, 6), None)

        if ret2 == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), critera)
            imgpoints.append(corners2)

        # cv.imshow('Video', frame)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break
finally:
    print("FINISHED processing")
    # input_video.release()
    # cv.destroyAllWindows()

print("(w, h)", (w, h))

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

# print("ret: ", ret)
print("camera matrix: ", mtx)
print("distortion coefficients: ", dist)
# print("rvec: ", rvecs)
# print("tvec: ", tvecs)

# Reset video capture to the beginning
input_video.set(cv.CAP_PROP_POS_FRAMES, 0)
fps = input_video.get(cv.CAP_PROP_FPS)
width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter Object
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output_undistorted.mp4', fourcc, fps, (width, height))

# Get optimal new camera matrix
newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

frame_count = 0

print(input_video.isOpened())
while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break

    frame_count += 1
    
    # Undistort image
    undistorted = cv.undistort(frame, mtx, dist, None, newCameraMtx)

    # Crop the frame if needed
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    if undistorted.shape[0] < height or undistorted.shape[1] < width:
        undistorted = cv.resize(undistorted, (width, height))

    out.write(undistorted)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")
input_video.release()
out.release()
cv.destroyAllWindows()
