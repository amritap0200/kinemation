import cv2
import numpy as np
from ultralytics import YOLO

CONNECTIONS = [
    ("left_wrist",  "left_elbow"),
    ("right_wrist", "right_elbow"),
    ("right_elbow", "neck"),
    ("left_elbow",  "neck"),
    ("neck",        "torso"),
    ("torso",       "left_knee"),
    ("torso",       "right_knee"),
    ("left_knee",   "left_ankle"),
    ("right_knee",  "right_ankle"),
    ("nose",        "neck"),
    ("nose",        "left_eye"),
    ("nose",        "right_eye"),
]

COLORS = [
    (0,   255, 0),
    (255, 0,   0),
    (0,   0,   255),
    (255, 165, 0),
    (128, 0,   128),
    (0,   255, 255),
]

def render_raw(video_path="trial1.mp4", output_path="output_raw_trial1.mp4"):
    model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    size = (1440, 810)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 5 or fps > 120:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    frame_count = 0
    print(f"Rendering RAW output...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, size)
        canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

        results = model(frame, verbose=False)[0]

        if results.keypoints is not None:
            keypoints = results.keypoints.xy.cpu().numpy()

            for idx, person in enumerate(keypoints):
                color = COLORS[idx % len(COLORS)]

                ls = (int(person[5][0]),  int(person[5][1]))
                rs = (int(person[6][0]),  int(person[6][1]))
                lh = (int(person[11][0]), int(person[11][1]))
                rh = (int(person[12][0]), int(person[12][1]))

                pts = {
                    "nose":        (int(person[0][0]),  int(person[0][1])),
                    "left_eye":    (int(person[1][0]),  int(person[1][1])),
                    "right_eye":   (int(person[2][0]),  int(person[2][1])),
                    "left_elbow":  (int(person[7][0]),  int(person[7][1])),
                    "right_elbow": (int(person[8][0]),  int(person[8][1])),
                    "left_wrist":  (int(person[9][0]),  int(person[9][1])),
                    "right_wrist": (int(person[10][0]), int(person[10][1])),
                    "left_knee":   (int(person[13][0]), int(person[13][1])),
                    "right_knee":  (int(person[14][0]), int(person[14][1])),
                    "left_ankle":  (int(person[15][0]), int(person[15][1])),
                    "right_ankle": (int(person[16][0]), int(person[16][1])),
                }

                pts = {k: v for k, v in pts.items() if v[0] > 0 and v[1] > 0}

                if ls[0] > 0 and rs[0] > 0:
                    pts["neck"] = (
                        (ls[0] + rs[0]) // 2,
                        (ls[1] + rs[1]) // 2
                    )
                if ls[0] > 0 and rs[0] > 0 and lh[0] > 0 and rh[0] > 0:
                    pts["torso"] = (
                        (ls[0] + rs[0] + lh[0] + rh[0]) // 4,
                        (ls[1] + rs[1] + lh[1] + rh[1]) // 4
                    )

                for point in pts.values():
                    cv2.circle(canvas, point, 4, color, -1)

                for p1, p2 in CONNECTIONS:
                    if p1 in pts and p2 in pts:
                        cv2.line(canvas, pts[p1], pts[p2], color, 2)

        cv2.putText(canvas, "RAW - No Temporal Smoothing",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        out.write(canvas)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Rendered {frame_count} frames...")

    cap.release()
    out.release()
    print(f"\nDone! Saved {frame_count} frames to {output_path}")

if __name__ == "__main__":
    render_raw()