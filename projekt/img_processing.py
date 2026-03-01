import cv2
import numpy as np
import os

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

def draw_stick_figure(img_processed, landmarks):

    image_h, image_w = img_processed.shape[:2]

    # Create a blank black canvas, same size as the processed image
    canvas = np.zeros((image_h, image_w, 3), dtype=np.uint8)

    # Calculate exact pixel coordinates from the normalized MediaPipe points
    pixel_points = []
    for landmark in landmarks:
        pixel_x = int(landmark.x * image_w)
        pixel_y = int(landmark.y * image_h)
        pixel_points.append((pixel_x, pixel_y))

    # Draw the joints 
    for point in pixel_points:
        cv2.circle(canvas, point, 8, (0, 255, 255), -1)

    # Draw the bones
    if len(pixel_points) >= 4:
        cv2.line(canvas, pixel_points[0], pixel_points[1], (0, 255, 0), 3) 
        cv2.line(canvas, pixel_points[0], pixel_points[2], (0, 255, 0), 3) 
        cv2.line(canvas, pixel_points[2], pixel_points[3], (0, 255, 0), 3) 

    # We stack them horizontally to easily compare
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
        # Generate dummy landmarks
        test_landmarks = [
            FakeLandmark(0.5, 0.2),  
            FakeLandmark(0.4, 0.4),  
            FakeLandmark(0.6, 0.4),  
            FakeLandmark(0.6, 0.6),  
        ]

        # Test drawing
        stick_canvas, comp_img = draw_stick_figure(processed, test_landmarks)
        
        # Test saving and displaying
        save_and_display(stick_canvas, comp_img, out_path)
