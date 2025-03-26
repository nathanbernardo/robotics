import freenect
import numpy as np
import cv2 as cv
from freenect import VIDEO_IR_10BIT

from kinect_utils.frame_convert import pretty_depth_cv

def get_depth():
    array = pretty_depth_cv(freenect.sync_get_depth(format=freenect.VIDEO_IR_8BIT)[0])
    print("ARRAY: ", array)
    return array


def main():
    while True:
        print(get_depth())
        cv.imshow('IR MAP', get_depth()) 

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
