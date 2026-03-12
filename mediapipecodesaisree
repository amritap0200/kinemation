import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

k=[0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
c=[(11, 12),(11, 13),(13, 15),(12, 14),(14, 16),(11, 23),(12, 24),(23, 24),(23, 25),(25, 27),(24, 26),(26, 28),(0, 11),(0, 12),]
b=python.BaseOptions(model_asset_path="pose_landmarker_heavy.task")
options = vision.PoseLandmarkerOptions(base_options=b,
                                    running_mode=vision.RunningMode.VIDEO,
                                    num_poses=5,min_pose_detection_confidence=0.4,
                                    min_pose_presence_confidence=0.6,min_tracking_confidence=0.7,)
detector=vision.PoseLandmarker.create_from_options(options)
cap=cv2.VideoCapture("trial1.mp4")
fps=cap.get(cv2.CAP_PROP_FPS)
count=0

while cap.isOpened():
    r, frame = cap.read()
    if not r:
        break
    height, width, _=frame.shape
    t=int((count/fps)*1000)
    e=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result=detector.detect_for_video(e, t)

    if result.pose_landmarks:
        for o in result.pose_landmarks:
            def get_point(d):
                lm=o[d]
                return (int(lm.x * width), int(lm.y * height))
            for a, b in c:
                cv2.line(frame, get_point(a), get_point(b), (200, 255, 0), 1)
                for l in k:
                    cv2.circle(frame, get_point(l), 4, (0, 120, 255), -1)

        cv2.imshow("TRIAL", frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        count += 1

cap.release()
cv2.destroyAllWindows()
