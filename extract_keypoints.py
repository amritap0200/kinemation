import cv2
import numpy as np
from ultralytics import YOLO

def extract_yolo_keypoints(video_path="test_vid.mp4", output_npy="yolo_keypoints.npy"):
    model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path} | Frames: {total_frames} | FPS: {fps} | Size: {width}x{height}")

    # First pass — figure out the maximum number of people detected in any frame
    print("Pass 1: counting max people per frame...")
    max_people = 0
    cap2 = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame_r = cv2.resize(frame, (1440, 810))
        results = model(frame_r, verbose=False)[0]
        if results.keypoints is not None:
            n = len(results.keypoints.xy)
            if n > max_people:
                max_people = n
    cap2.release()
    print(f"Max people in any frame: {max_people}")

    if max_people == 0:
        print("ERROR: No people detected in video at all.")
        return None

    # Second pass — extract keypoints, shape will be (N, max_people, 17, 2)
    # If fewer than max_people detected in a frame, fill missing slots with zeros
    print("Pass 2: extracting keypoints...")
    all_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_r = cv2.resize(frame, (1440, 810))
        results = model(frame_r, verbose=False)[0]

        frame_data = np.zeros((max_people, 17, 2), dtype=np.float32)

        if results.keypoints is not None:
            kps = results.keypoints.xy.cpu().numpy()  # (num_detected, 17, 2)
            for p in range(min(len(kps), max_people)):
                frame_data[p] = kps[p]

        all_frames.append(frame_data)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    cap.release()

    # Shape: (N, max_people, 17, 2)
    keypoints_array = np.array(all_frames, dtype=np.float32)
    np.save(output_npy, keypoints_array)
    print(f"\nDone! Shape: {keypoints_array.shape}  (frames x people x joints x coords)")
    return keypoints_array

if __name__ == "__main__":
    extract_yolo_keypoints(video_path="trial1.mp4", output_npy="yolo_keypoints_trial1.npy")