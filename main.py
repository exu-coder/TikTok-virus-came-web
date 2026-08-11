import cv2
import numpy as np
import mediapipe as mp
import math
import random
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Filter states
FILTERS = [
    "PORTAL_SKETCH",
    "PORTAL_GLITCH",
    "PORTAL_NEON",
    "PORTAL_INVERT",
    "PORTAL_PIXELATE",
    "PORTAL_WARP",
    "PORTAL_THERMAL",
    "PORTAL_MIRROR",
    "PORTAL_ZOOM",
    "PORTAL_BLUR",
]

current_filter_idx = 0
filter_change_cooldown = 0

# Glitch effect history
glitch_buffer = deque(maxlen=5)

def get_hand_rectangle(image, landmarks):
    """Get bounding box from hand landmarks"""
    h, w = image.shape[:2]
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    padding = 30
    x_min = max(0, int(x_min - padding))
    y_min = max(0, int(y_min - padding))
    x_max = min(w, int(x_max + padding))
    y_max = min(h, int(y_max + padding))

    return (x_min, y_min, x_max, y_max)

def get_finger_frame(image, landmarks):
    """
    Create a frame between thumb tip (4) and pinky tip (20)
    with top frame using index tip (8) and middle tip (12)
    """
    h, w = image.shape[:2]

    thumb = landmarks.landmark[4]
    index = landmarks.landmark[8]
    middle = landmarks.landmark[12]
    ring = landmarks.landmark[16]
    pinky = landmarks.landmark[20]
    wrist = landmarks.landmark[0]

    thumb_pt = (int(thumb.x * w), int(thumb.y * h))
    index_pt = (int(index.x * w), int(index.y * h))
    middle_pt = (int(middle.x * w), int(middle.y * h))
    ring_pt = (int(ring.x * w), int(ring.y * h))
    pinky_pt = (int(pinky.x * w), int(pinky.y * h))
    wrist_pt = (int(wrist.x * w), int(wrist.y * h))

    hand_width = math.dist(thumb_pt, pinky_pt)
    hand_height = math.dist(wrist_pt, middle_pt)

    center_x = (thumb_pt[0] + pinky_pt[0]) // 2
    center_y = (thumb_pt[1] + pinky_pt[1]) // 2

    portal_w = int(hand_width * 1.5)
    portal_h = int(hand_height * 2.5)

    angle = math.atan2(pinky_pt[1] - thumb_pt[1], pinky_pt[0] - thumb_pt[0])

    cos_a, sin_a = math.cos(angle), math.sin(angle)

    dx = portal_w // 2
    dy = portal_h // 2

    corners = [
        (center_x - dx * cos_a - dy * sin_a, center_y - dx * sin_a + dy * cos_a),
        (center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a),
        (center_x + dx * cos_a + dy * sin_a, center_y + dx * sin_a - dy * cos_a),
        (center_x - dx * cos_a + dy * sin_a, center_y - dx * sin_a - dy * cos_a),
    ]

    corners = [(int(x), int(y)) for x, y in corners]

    return corners, (center_x, center_y), portal_w, portal_h

def apply_sketch_effect(frame):
    """Pencil sketch effect"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_glitch_effect(frame):
    """RGB channel shift glitch"""
    b, g, r = cv2.split(frame)

    shift_x = random.randint(-10, 10)
    shift_y = random.randint(-5, 5)

    M_b = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    M_r = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])

    b_shifted = cv2.warpAffine(b, M_b, (frame.shape[1], frame.shape[0]))
    r_shifted = cv2.warpAffine(r, M_r, (frame.shape[1], frame.shape[0]))

    glitched = cv2.merge([b_shifted, g, r_shifted])

    for i in range(0, frame.shape[0], 4):
        cv2.line(glitched, (0, i), (frame.shape[1], i), (0, 0, 0), 1)

    return glitched

def apply_neon_effect(frame):
    """Neon edge glow effect"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    neon = np.zeros_like(frame)
    neon[edges > 0] = (0, 255, 255)

    glow = cv2.GaussianBlur(neon, (15, 15), 0)

    darkened = cv2.multiply(frame, np.array([0.3, 0.3, 0.3]))
    result = cv2.add(darkened, glow)

    return result

def apply_invert_effect(frame):
    """Color inversion with enhanced contrast"""
    inverted = cv2.bitwise_not(frame)
    blue_tint = np.zeros_like(frame)
    blue_tint[:, :] = (100, 50, 0)
    result = cv2.addWeighted(inverted, 0.8, blue_tint, 0.2, 0)
    return result

def apply_pixelate_effect(frame):
    """Pixelation effect"""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 20, h // 20), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def apply_warp_effect(frame, time_val):
    """Wave distortion effect"""
    h, w = frame.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    for y in range(h):
        for x in range(w):
            offset_x = int(20 * math.sin(2 * math.pi * y / 60 + time_val))
            offset_y = int(10 * math.cos(2 * math.pi * x / 80 + time_val))
            map_x[y, x] = min(w - 1, max(0, x + offset_x))
            map_y[y, x] = min(h - 1, max(0, y + offset_y))

    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    return warped

def apply_thermal_effect(frame):
    """False color thermal camera look"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

def apply_mirror_effect(frame):
    """Split mirror effect"""
    h, w = frame.shape[:2]
    left = frame[:, :w//2]
    right = cv2.flip(left, 1)
    mirrored = np.hstack([left, right])
    return cv2.resize(mirrored, (w, h))

def apply_zoom_effect(frame, center):
    """Zoom burst from center"""
    h, w = frame.shape[:2]
    cx, cy = center

    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx**2 + dy**2) + 1
            factor = 1 + dist / 200
            map_x[y, x] = cx + dx / factor
            map_y[y, x] = cy + dy / factor

    zoomed = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    return zoomed

def apply_blur_effect(frame):
    """Radial motion blur"""
    kernel_size = 15
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    return cv2.addWeighted(frame, 0.6, blurred, 0.4, 0)

def apply_filter(frame, filter_name, center, time_val):
    """Apply the selected filter"""
    if filter_name == "PORTAL_SKETCH":
        return apply_sketch_effect(frame)
    elif filter_name == "PORTAL_GLITCH":
        return apply_glitch_effect(frame)
    elif filter_name == "PORTAL_NEON":
        return apply_neon_effect(frame)
    elif filter_name == "PORTAL_INVERT":
        return apply_invert_effect(frame)
    elif filter_name == "PORTAL_PIXELATE":
        return apply_pixelate_effect(frame)
    elif filter_name == "PORTAL_WARP":
        return apply_warp_effect(frame, time_val)
    elif filter_name == "PORTAL_THERMAL":
        return apply_thermal_effect(frame)
    elif filter_name == "PORTAL_MIRROR":
        return apply_mirror_effect(frame)
    elif filter_name == "PORTAL_ZOOM":
        return apply_zoom_effect(frame, center)
    elif filter_name == "PORTAL_BLUR":
        return apply_blur_effect(frame)
    return frame

def create_portal_mask(corners, shape):
    """Create a mask for the portal area"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(corners, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def warp_perspective_to_portal(frame, corners, portal_w, portal_h):
    """Warp frame to fit the portal shape"""
    dst_pts = np.array([
        [0, 0],
        [portal_w, 0],
        [portal_w, portal_h],
        [0, portal_h]
    ], np.float32)

    src_pts = np.array(corners, np.float32)
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)

    warped = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
    return warped

def main():
    global current_filter_idx, filter_change_cooldown

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    time_val = 0

    print("Controls:")
    print("  SPACE - Change filter")
    print("  Q - Quit")
    print("Show your hand to the camera!")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        output = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                corners, center, portal_w, portal_h = get_finger_frame(frame, hand_landmarks)

                if portal_w > 50 and portal_h > 50 and portal_w < w and portal_h < h:
                    mask = create_portal_mask(corners, frame.shape)

                    current_filter = FILTERS[current_filter_idx]

                    filtered_frame = apply_filter(frame, current_filter, center, time_val)

                    warped_filtered = warp_perspective_to_portal(filtered_frame, corners, portal_w, portal_h)

                    inv_mask = cv2.bitwise_not(mask)

                    outside = cv2.bitwise_and(output, output, mask=inv_mask)
                    inside = cv2.bitwise_and(warped_filtered, warped_filtered, mask=mask)

                    output = cv2.add(outside, inside)

                    cv2.polylines(output, [np.array(corners)], True, (255, 255, 255), 2)

                    for corner in corners:
                        cv2.circle(output, corner, 5, (0, 255, 0), -1)

                    label = current_filter.replace("PORTAL_", "")
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_x = corners[0][0]
                    label_y = corners[0][1] - 10

                    cv2.rectangle(output, 
                                (label_x, label_y - label_size[1] - 5),
                                (label_x + label_size[0], label_y + 5),
                                (0, 0, 0), -1)
                    cv2.putText(output, label, (label_x, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(output, "Hand Tracking AR Portal", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"Filter: {FILTERS[current_filter_idx].replace('PORTAL_', '')}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, "SPACE: Change | Q: Quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Hand Tracking AR Portal', output)

        time_val += 0.1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            current_filter_idx = (current_filter_idx + 1) % len(FILTERS)
            print(f"Filter: {FILTERS[current_filter_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
