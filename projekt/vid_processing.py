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

def split_video_to_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return [], 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frames.append(frame)

    cap.release()
    print(f"Extracted {frame_num}/{total_frames} frames at {fps} FPS")
    return frames, fps

def preprocess_frame(frame, max_dim=800):

    original_h, original_w = frame.shape[:2]

    # 1. Resizing
    if max(original_h, original_w) > max_dim:
        scale = max_dim / max(original_h, original_w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 2. Noise Reduction (gaussian blur)
    frame_processed = cv2.GaussianBlur(frame, (5, 5), 0)

    # 3. Improved Contrast with CLAHE
    # (Uncomment this block to enable CLAHE)
    # lab = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # l = clahe.apply(l)
    # lab = cv2.merge([l, a, b])
    # frame_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return frame_processed, frame

def draw_stick_figure(frame_processed, landmarks, color=(0, 255, 0), canvas=None):

    image_h, image_w = frame_processed.shape[:2]

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

    comparison = np.hstack([frame_processed, canvas])

    return canvas, comparison

def process_video(video_path, output_path, get_landmarks_func=None):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Preprocess one frame to figure out the output dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        cap.release()
        return
    test_processed, _ = preprocess_frame(first_frame)
    out_h, out_w = test_processed.shape[:2]

    # Reset video back to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        print(f"Processing frame {frame_num}/{total_frames}...")

        # Preprocess this frame
        processed, _ = preprocess_frame(frame)

        # Get landmarks for this frame
        if get_landmarks_func is not None:
            landmarks = get_landmarks_func(processed)
        else:
            landmarks = generate_dummy_landmarks()

        # Draw stick figure
        canvas, _ = draw_stick_figure(processed, landmarks)

        # Write the stick figure frame to output video
        out.write(canvas)

    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")

def generate_dummy_landmarks():
    return [
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


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_vid_path = os.path.join(script_dir, 'datasets', 'test_video.mp4')
    out_path = os.path.join(script_dir, 'datasets', 'stick_figure_output.mp4')

    process_video(test_vid_path, out_path)
