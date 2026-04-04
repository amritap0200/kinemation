"""
3D Pose Visualization Module

Provides rendering of 3D skeletons with depth coloring.
"""

import cv2
import numpy as np
from mediapipe_to_h36m import H36M_CONNECTIONS, H36M_JOINT_NAMES


# Color palette for multiple people
PERSON_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue (BGR)
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 255, 128),  # Spring green
    (128, 255, 0),  # Chartreuse
]


def depth_to_color(z_normalized, base_color):
    """
    Modify color based on depth (z coordinate).
    Closer points are brighter, farther points are darker.
    
    Args:
        z_normalized: depth value normalized to [-1, 1] range
        base_color: (B, G, R) tuple
        
    Returns:
        Modified color tuple
    """
    # Map z to brightness factor [0.4, 1.0]
    # Negative z (closer to camera) = brighter
    brightness = 0.7 - 0.3 * z_normalized
    brightness = np.clip(brightness, 0.4, 1.0)
    
    return tuple(int(c * brightness) for c in base_color)


def draw_3d_skeleton(canvas, keypoints_3d, color=(0, 255, 0), 
                     video_width=None, video_height=None,
                     draw_depth_colors=True):
    """
    Draw a 3D skeleton projected to 2D with depth coloring.
    
    Args:
        canvas: numpy array (H, W, 3) to draw on
        keypoints_3d: array of shape (17, 3) with (x, y, z) in normalized coords
        color: base color for this person
        video_width: width for denormalization (None = use canvas width)
        video_height: height for denormalization (None = use canvas height)
        draw_depth_colors: whether to color by depth
        
    Returns:
        canvas with skeleton drawn
    """
    h, w = canvas.shape[:2]
    video_width = video_width or w
    video_height = video_height or h
    
    if keypoints_3d is None or keypoints_3d.max() == 0:
        return canvas
    
    # Convert normalized coordinates to pixel coordinates
    # VideoPose3D outputs normalized coords where x,y are in [-1, 1]
    points_2d = []
    z_values = []
    
    for i in range(17):
        x, y, z = keypoints_3d[i]
        
        # Skip invalid joints
        if x == 0 and y == 0:
            points_2d.append(None)
            z_values.append(0)
            continue
        
        # Denormalize x, y (from [-1, 1] to pixel coords)
        px = int((x + 1) * video_width / 2)
        py = int((y + 1) * video_height / 2)
        
        # Clamp to canvas bounds
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        
        points_2d.append((px, py))
        z_values.append(z)
    
    # Normalize z values for coloring
    valid_z = [z for z in z_values if z != 0]
    if valid_z:
        z_min, z_max = min(valid_z), max(valid_z)
        z_range = z_max - z_min if z_max != z_min else 1.0
    else:
        z_min, z_range = 0, 1.0
    
    # Draw connections (bones)
    for idx_a, idx_b in H36M_CONNECTIONS:
        if points_2d[idx_a] is not None and points_2d[idx_b] is not None:
            if draw_depth_colors:
                # Average depth for the bone
                avg_z = (z_values[idx_a] + z_values[idx_b]) / 2
                z_norm = (avg_z - z_min) / z_range * 2 - 1  # Map to [-1, 1]
                line_color = depth_to_color(z_norm, color)
            else:
                line_color = color
            
            cv2.line(canvas, points_2d[idx_a], points_2d[idx_b], line_color, 2)
    
    # Draw joints
    for i, point in enumerate(points_2d):
        if point is not None:
            if draw_depth_colors:
                z_norm = (z_values[i] - z_min) / z_range * 2 - 1
                joint_color = depth_to_color(z_norm, color)
            else:
                joint_color = color
            
            # Vary joint size by depth (closer = larger)
            radius = int(4 - z_norm) if draw_depth_colors else 4
            radius = np.clip(radius, 3, 6)
            cv2.circle(canvas, point, radius, joint_color, -1)
    
    return canvas


def draw_2d_skeleton_h36m(canvas, keypoints_2d, color=(0, 255, 0)):
    """
    Draw a 2D skeleton using H36M 17-joint format.
    
    Args:
        canvas: numpy array (H, W, 3) to draw on
        keypoints_2d: array of shape (17, 2) with (x, y) in pixel coords
        color: color for this person
        
    Returns:
        canvas with skeleton drawn
    """
    if keypoints_2d is None or keypoints_2d.max() == 0:
        return canvas
    
    h, w = canvas.shape[:2]
    
    # Draw connections
    for idx_a, idx_b in H36M_CONNECTIONS:
        pt_a = keypoints_2d[idx_a]
        pt_b = keypoints_2d[idx_b]
        
        if pt_a[0] > 0 and pt_a[1] > 0 and pt_b[0] > 0 and pt_b[1] > 0:
            pt_a = (int(pt_a[0]), int(pt_a[1]))
            pt_b = (int(pt_b[0]), int(pt_b[1]))
            cv2.line(canvas, pt_a, pt_b, color, 2)
    
    # Draw joints
    for i, point in enumerate(keypoints_2d):
        if point[0] > 0 and point[1] > 0:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 4, color, -1)
    
    return canvas


class Visualizer3D:
    """
    Visualizer for 3D pose estimation output.
    """
    
    def __init__(self, video_width, video_height, max_people=10):
        self.video_width = video_width
        self.video_height = video_height
        self.max_people = max_people
    
    def render_frame(self, keypoints_3d_all, mode='skeleton', background=None):
        """
        Render one frame of 3D poses.
        
        Args:
            keypoints_3d_all: array of shape (max_people, 17, 3)
            mode: 'skeleton' (black bg), 'overlay' (on video frame)
            background: video frame for overlay mode
            
        Returns:
            rendered frame
        """
        if mode == 'overlay' and background is not None:
            canvas = background.copy()
        else:
            canvas = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        for person_id in range(self.max_people):
            if keypoints_3d_all[person_id].max() == 0:
                continue
            
            color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
            canvas = draw_3d_skeleton(
                canvas, 
                keypoints_3d_all[person_id],
                color=color,
                video_width=self.video_width,
                video_height=self.video_height,
                draw_depth_colors=True
            )
        
        return canvas
    
    def render_side_by_side(self, keypoints_2d_all, keypoints_3d_all, 
                            original_frame=None, labels=True):
        """
        Render 2D and 3D poses side by side.
        
        Args:
            keypoints_2d_all: array of shape (max_people, 17, 2)
            keypoints_3d_all: array of shape (max_people, 17, 3)
            original_frame: optional video frame
            labels: whether to add text labels
            
        Returns:
            combined frame (width * 2)
        """
        # Left: 2D skeleton (or original with overlay)
        if original_frame is not None:
            left = original_frame.copy()
        else:
            left = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        for person_id in range(self.max_people):
            if keypoints_2d_all[person_id].max() == 0:
                continue
            color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
            left = draw_2d_skeleton_h36m(left, keypoints_2d_all[person_id], color)
        
        # Right: 3D skeleton
        right = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        for person_id in range(self.max_people):
            if keypoints_3d_all[person_id].max() == 0:
                continue
            color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
            right = draw_3d_skeleton(
                right, 
                keypoints_3d_all[person_id],
                color=color,
                video_width=self.video_width,
                video_height=self.video_height
            )
        
        if labels:
            cv2.putText(left, "2D Pose", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(right, "3D Pose (depth colored)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return np.hstack([left, right])


def create_depth_colorbar(height=400, width=30):
    """
    Create a vertical colorbar showing depth mapping.
    
    Returns:
        colorbar image
    """
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        z_norm = (y / height) * 2 - 1  # Top = close, bottom = far
        color = depth_to_color(-z_norm, (0, 255, 0))  # Negate for visual
        colorbar[y, :] = color
    
    return colorbar
