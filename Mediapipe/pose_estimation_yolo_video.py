import cv2
import numpy as np
from ultralytics import YOLO


def yolo_pose(source="test_vid.mp4", size=(1440, 810)):

    model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error opening video")
        return

    width, height = size

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 5 or fps > 120:
        fps = 30

    print("FPS:", fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    if not out.isOpened():
        print("VideoWriter failed")
        return

    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, size)

        results = model(frame, verbose=False)[0]

        if results.keypoints is not None:

            keypoints = results.keypoints.xy.cpu().numpy()

            for idx, person in enumerate(keypoints):

                color = colors[idx % len(colors)]

                # YOLO keypoints → dictionary
                pts = {
                    "nose": (int(person[0][0]), int(person[0][1])),
                    "left_eye": (int(person[1][0]), int(person[1][1])),
                    "right_eye": (int(person[2][0]), int(person[2][1])),
                    "left_elbow": (int(person[7][0]), int(person[7][1])),
                    "right_elbow": (int(person[8][0]), int(person[8][1])),
                    "left_wrist": (int(person[9][0]), int(person[9][1])),
                    "right_wrist": (int(person[10][0]), int(person[10][1])),
                    "left_knee": (int(person[13][0]), int(person[13][1])),
                    "right_knee": (int(person[14][0]), int(person[14][1])),
                    "left_ankle": (int(person[15][0]), int(person[15][1])),
                    "right_ankle": (int(person[16][0]), int(person[16][1])),
                    
                }
                lh= (int(person[11][0]), int(person[11][1]))
                rh= (int(person[12][0]), int(person[12][1]))
                ls = (int(person[5][0]), int(person[5][1]))
                rs = (int(person[6][0]), int(person[6][1]))
                pts = {k: v for k, v in pts.items() if v[0] > 0 and v[1] > 0}

                
                neck = (
                        (ls[0] + rs[0]) // 2,
                        (ls[1] + rs[1]) // 2
                    )
                pts["neck"] = neck

                
                torso = (
                        (ls[0] + rs[0] + lh[0] + rh[0]) // 4,
                        (ls[1] + rs[1] + lh[1] + rh[1]) // 4
                    )
                pts["torso"] = torso

                for point in pts.values():
                    cv2.circle(frame, point, 4, color, -1)

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
                    if p1 in pts and p2 in pts:
                        cv2.line(frame, pts[p1], pts[p2], color, 2)

        out.write(frame)

        cv2.imshow("YOLO Custom Pose", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_pose("test_vid.mp4")