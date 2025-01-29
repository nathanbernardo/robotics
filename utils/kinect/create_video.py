
#!/usr/bin/env python
import freenect
import cv2
from kinect_utils.frame_convert import pretty_depth_cv, video_cv

print('Press ESC in window to stop')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (640, 480))

def get_video():
    return video_cv(freenect.sync_get_video()[0])

while 1:
    frame = get_video()

    out.write(frame)

    cv2.imshow('Kinect Video', frame)
    if cv2.waitKey(10) == 27:
        break

out.release()
cv2.destroyAllWindows()

