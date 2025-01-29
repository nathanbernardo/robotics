import numpy as np
import cv2 as cv
import glob

critera = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7,0:6].T.reshape(-1, 2)


objpoints = []
imgpoints = []

images = glob.glob('*.jpg')
h, w = 0, 0
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), critera)
        imgpoints.append(corners2)

        # cv.drawChessboardCorners(img, (7,6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)

# cv.destroyAllWindows()

# calibrate camera
print("(w, h): ", (w, h))
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print("DIST: ", dist)
img = cv.imread('left12.jpg')
h , w = img.shape[:2]
newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print("new cmaera matrix: ", newCameraMtx)
dst = cv.undistort(img, mtx, dist, None, newCameraMtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibrate.png', dst)


