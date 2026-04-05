import cv2
import mediapipe as mp
from ultralytics import YOLO

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

yolo = YOLO("yolov8m.pt")

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_full.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(options)

image = cv2.imread("test7.jpeg")
h, w, _ = image.shape

results = yolo(image)[0]

boxes = []
for box in results.boxes:
    cls = int(box.cls[0])
    if cls == 0:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2))

def to_global(lm, x_offset, y_offset, crop_w, crop_h):
    x = int(lm.x * crop_w + x_offset)
    y = int(lm.y * crop_h + y_offset)
    return (x, y)

colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]

for idx, (x1, y1, x2, y2) in enumerate(boxes):

    crop = image[y1:y2, x1:x2]
    crop_h, crop_w, _ = crop.shape

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

    result = detector.detect(mp_image)

    color = colors[idx % len(colors)]

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        pixel_points = {
            "nose": to_global(lm[0], x1, y1, crop_w, crop_h),
            "left_eye": to_global(lm[2], x1, y1, crop_w, crop_h),
            "right_eye": to_global(lm[5], x1, y1, crop_w, crop_h),
            "left_wrist": to_global(lm[15], x1, y1, crop_w, crop_h),
            "right_wrist": to_global(lm[16], x1, y1, crop_w, crop_h),
            "left_elbow": to_global(lm[13], x1, y1, crop_w, crop_h),
            "right_elbow": to_global(lm[14], x1, y1, crop_w, crop_h),
            "left_knee": to_global(lm[25], x1, y1, crop_w, crop_h),
            "right_knee": to_global(lm[26], x1, y1, crop_w, crop_h),
            "left_ankle": to_global(lm[27], x1, y1, crop_w, crop_h),
            "right_ankle": to_global(lm[28], x1, y1, crop_w, crop_h)
        }

        ls = lm[11]
        rs = lm[12]
        lh = lm[23]
        rh = lm[24]

        neck_x = (ls.x + rs.x) / 2
        neck_y = (ls.y + rs.y) / 2
        pixel_points["neck"] = to_global(
            type("obj", (), {"x": neck_x, "y": neck_y}),
            x1, y1, crop_w, crop_h
        )

        torso_x = (ls.x + rs.x + lh.x + rh.x) / 4
        torso_y = (ls.y + rs.y + lh.y + rh.y) / 4
        pixel_points["torso"] = to_global(
            type("obj", (), {"x": torso_x, "y": torso_y}),
            x1, y1, crop_w, crop_h
        )

        for point in pixel_points.values():
            cv2.circle(image, point, 5, color, -1)

        connections = [
            ("left_wrist", "left_elbow"),
            ("right_wrist", "right_elbow"),
            ("right_elbow", "neck"),
            ("left_elbow", "neck"),
            ("neck", "torso"),
            ("torso", "left_knee"),
            ("torso", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
            ("nose", "neck"),
            ("nose", "left_eye"),
            ("nose", "right_eye")
        ]

        for p1, p2 in connections:
            if p1 in pixel_points and p2 in pixel_points:
                cv2.line(image, pixel_points[p1], pixel_points[p2], color, 2)

cv2.imwrite("output.jpeg", image)
cv2.imshow("YOLO + MediaPipe Pose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()