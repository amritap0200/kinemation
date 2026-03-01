import cv2
import numpy as np
import os

POSE_PAIRS = [
    [0, 1],   [1, 2],   [2, 3],   [3, 7],
    [0, 4],   [4, 5],   [5, 6],   [6, 8],
    [9, 10],
    [11, 12],
    [11, 13], [13, 15],
    [12, 14], [14, 16],
    [15, 17], [15, 19], [15, 21],
    [16, 18], [16, 20], [16, 22],
    [23, 24],
    [23, 25], [25, 27],
    [24, 26], [26, 28],
    [27, 29], [29, 31],
    [28, 30], [30, 32],
]

PERSON_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
    (128, 255, 0),
]

class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def preprocess_image(img_path, max_dim=800):

    img = cv2.imread(img_path)

    original_h, original_w = img.shape[:2]

    # 1. Resizing
    if max(original_h, original_w) > max_dim:
        scale = max_dim / max(original_h, original_w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Resized to: {img.shape[1]} x {img.shape[0]}")

    # 2. Noise Reduction (gaussian blur)
    img_processed = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Improved Contrast with CLAHE
    # (Uncomment this block to enable CLAHE)
    # lab = cv2.cvtColor(img_processed, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # l = clahe.apply(l)
    # lab = cv2.merge([l, a, b])
    # img_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img_processed, img

def draw_stick_figure(img_processed, landmarks, color=(0, 255, 0), canvas=None):

    image_h, image_w = img_processed.shape[:2]

    if canvas is None:
        canvas = np.zeros((image_h, image_w, 3), dtype=np.uint8)

    pixel_points = []
    for landmark in landmarks:
        pixel_x = int(landmark.x * image_w)
        pixel_y = int(landmark.y * image_h)
        pixel_points.append((pixel_x, pixel_y))

    for pair in POSE_PAIRS:
        idx_a, idx_b = pair
        if idx_a < len(pixel_points) and idx_b < len(pixel_points):
            if pixel_points[idx_a] is not None and pixel_points[idx_b] is not None:
                cv2.line(canvas, pixel_points[idx_a], pixel_points[idx_b], color, 2)

    if len(pixel_points) >= 25:
        mid_shoulder = (
            (pixel_points[11][0] + pixel_points[12][0]) // 2,
            (pixel_points[11][1] + pixel_points[12][1]) // 2
        )
        mid_hip = (
            (pixel_points[23][0] + pixel_points[24][0]) // 2,
            (pixel_points[23][1] + pixel_points[24][1]) // 2
        )
        cv2.line(canvas, mid_shoulder, mid_hip, color, 2)

    for point in pixel_points:
        cv2.circle(canvas, point, 4, color, -1)

    comparison = np.hstack([img_processed, canvas])

    return canvas, comparison

def save_and_display(canvas, comparison, output_path):

    cv2.imwrite(output_path, canvas)
    print(f"Stick figure saved to: {output_path}")

    cv2.imshow("Processed Image  |  Stick Figure", comparison)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_img_path = os.path.join(script_dir, 'datasets', 'person.jpg')
    out_path = os.path.join(script_dir, 'datasets', 'stick_figure_output.png')

    # Test processing
    processed, original = preprocess_image(test_img_path)
    
    if processed is not None:
        # These are fake positions that roughly form a stick figure
        test_landmarks = [
            FakeLandmark(0.50, 0.12),  # 0  nose
            FakeLandmark(0.51, 0.11),  # 1  left eye inner
            FakeLandmark(0.52, 0.11),  # 2  left eye
            FakeLandmark(0.53, 0.11),  # 3  left eye outer
            FakeLandmark(0.49, 0.11),  # 4  right eye inner
            FakeLandmark(0.48, 0.11),  # 5  right eye
            FakeLandmark(0.47, 0.11),  # 6  right eye outer
            FakeLandmark(0.54, 0.12),  # 7  left ear
            FakeLandmark(0.46, 0.12),  # 8  right ear
            FakeLandmark(0.51, 0.14),  # 9  mouth left
            FakeLandmark(0.49, 0.14),  # 10 mouth right
            FakeLandmark(0.58, 0.28),  # 11 left shoulder
            FakeLandmark(0.42, 0.28),  # 12 right shoulder
            FakeLandmark(0.64, 0.40),  # 13 left elbow
            FakeLandmark(0.36, 0.40),  # 14 right elbow
            FakeLandmark(0.68, 0.52),  # 15 left wrist
            FakeLandmark(0.32, 0.52),  # 16 right wrist
            FakeLandmark(0.70, 0.54),  # 17 left pinky
            FakeLandmark(0.30, 0.54),  # 18 right pinky
            FakeLandmark(0.69, 0.53),  # 19 left index
            FakeLandmark(0.31, 0.53),  # 20 right index
            FakeLandmark(0.71, 0.52),  # 21 left thumb
            FakeLandmark(0.29, 0.52),  # 22 right thumb
            FakeLandmark(0.55, 0.58),  # 23 left hip
            FakeLandmark(0.45, 0.58),  # 24 right hip
            FakeLandmark(0.56, 0.72),  # 25 left knee
            FakeLandmark(0.44, 0.72),  # 26 right knee
            FakeLandmark(0.57, 0.88),  # 27 left ankle
            FakeLandmark(0.43, 0.88),  # 28 right ankle
            FakeLandmark(0.58, 0.90),  # 29 left heel
            FakeLandmark(0.42, 0.90),  # 30 right heel
            FakeLandmark(0.59, 0.92),  # 31 left foot index
            FakeLandmark(0.41, 0.92),  # 32 right foot index
        ]

        # Test drawing
        stick_canvas, comp_img = draw_stick_figure(processed, test_landmarks)
        
        # Test saving and displaying
        save_and_display(stick_canvas, comp_img, out_path)
