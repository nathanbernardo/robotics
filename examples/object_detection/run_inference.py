import cv2
from ultralytics import YOLO
import random
from ultralytics.utils.plotting import Annotator, colors

# Load OBB model
model = YOLO("/home/ncbernar/.pyenv/runs/obb/train41/weights/best.pt")

# Generate a random color for each class
color_map = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in model.names}

input_video = cv2.VideoCapture("./IMG_7474.mov")

fps = int(input_video.get(cv2.CAP_PROP_FPS))
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

names = model.names

def draw_boxes_and_centers(results):
    annotated_frame = results[0].plot()
    for r in results:
        obbs = r.obb.xyxy.cpu().numpy().astype(int)
        # classes = r.boxes.cls.cpu().numpy().astype(int)

        for box in obbs:
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            cv2.circle(annotated_frame, (center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)

    return annotated_frame

while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break

    annotator = Annotator(
        frame,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False
    )

    # Run inference on the frame
    results = model(frame)
    boxes = results[0].obb.xyxy.cpu()
    clss = results[0].obb.cls.cpu().tolist()
    confs = results[0].obb.conf.cpu().tolist()

    for box, cls, conf in zip(boxes, clss, confs):
        label = f"{names[int(cls)]} {conf:.2f}"
        if names[int(cls)] == '8oz':
            center_x = int((box[0] + box[2]) // 2)
            center_y = int((box[1] + box[3]) // 2)
            print("x and y center: ", (center_x, center_y))
            annotator.box_label(box, label=label, color=colors(int(cls), True))
            cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=1)


    # Draw the results on the frame
    # annotated_frame = draw_boxes_and_centers(results)

    # Write the frame to the output video
    # output_video.write(annotated_frame)
    output_video.write(frame)
    cv2.imshow("Circle annotation", frame)

    # Optional: Display the frame (comment out if not needed)
    # cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
input_video.release()
output_video.release()
cv2.destroyAllWindows()

