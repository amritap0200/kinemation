import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

image = cv2.imread("test2.jpeg")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pose.process(rgb_image)

if results.pose_landmarks:
    lm = results.pose_landmarks.landmark
    h, w, _ = image.shape

    def to_pixel(p):
        return (int(p.x * w), int(p.y * h))

    pixel_points = {
        "nose": to_pixel(lm[mp_pose.PoseLandmark.NOSE]),
        "left_eye": to_pixel(lm[mp_pose.PoseLandmark.LEFT_EYE]),
        "right_eye": to_pixel(lm[mp_pose.PoseLandmark.RIGHT_EYE]),
        "left_wrist": to_pixel(lm[mp_pose.PoseLandmark.LEFT_WRIST]),
        "right_wrist": to_pixel(lm[mp_pose.PoseLandmark.RIGHT_WRIST]),
        "left_elbow": to_pixel(lm[mp_pose.PoseLandmark.LEFT_ELBOW]),
        "right_elbow": to_pixel(lm[mp_pose.PoseLandmark.RIGHT_ELBOW]),
        "left_knee": to_pixel(lm[mp_pose.PoseLandmark.LEFT_KNEE]),
        "right_knee": to_pixel(lm[mp_pose.PoseLandmark.RIGHT_KNEE]),
        "left_ankle": to_pixel(lm[mp_pose.PoseLandmark.LEFT_ANKLE]),
        "right_ankle": to_pixel(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
    }

    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

    neck_x = (ls.x + rs.x) / 2
    neck_y = (ls.y + rs.y) / 2
    pixel_points["neck"] = (int(neck_x * w), int(neck_y * h))

    torso_x = (ls.x + rs.x + lh.x + rh.x) / 4
    torso_y = ((ls.y + rs.y)/2 + (lh.y + rh.y)) / 3
    pixel_points["torso"] = (int(torso_x * w), int(torso_y * h))

    for point in pixel_points.values():
        cv2.circle(image, point, 5, (0, 255, 0), -1)

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
            cv2.line(image, pixel_points[p1], pixel_points[p2], (255, 255, 255), 2)

cv2.imwrite("output.png", image)

cv2.imshow("Custom Pose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
