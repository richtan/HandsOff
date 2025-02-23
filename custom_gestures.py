import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from mac_actions import volume_up, volume_down


# Model Gesture

last_gestures = []


def detect_transitions_and_features():
    global last_gestures

    if last_gestures[-1][0] == "Thumb_Up":
        volume_up(1)
        last_gestures.pop()
        return
    elif last_gestures[-1][0] == "Thumb_Down":
        volume_down(1)
        last_gestures.pop()
        return

    open_palm_time = None
    closed_fist_time = None

    for last_gesture in last_gestures:
        gesture, timestamp = last_gesture

        if gesture == "Open_Palm":
            open_palm_time = timestamp

            if closed_fist_time:
                time_diff = timestamp - closed_fist_time

                # Transition must happen within 1 second
                if 0 < time_diff <= 1000:
                    print("Gesture Transition Detected: Closed Fist → Open Fist")
                    # action here
                    last_gestures.clear()
                    break
        elif gesture == "Closed_Fist":
            closed_fist_time = timestamp

            if open_palm_time:
                time_diff = timestamp - open_palm_time

                # Transition must happen within 1 second
                if 0 < time_diff <= 1000:
                    print("Gesture Transition Detected: Open Palm → Closed Fist")
                    # action here
                    last_gestures.clear()
                    break


def gesture_result_callback(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    if result.gestures:
        # Get the category name of the recognized gesture
        category_name = result.gestures[0][0].category_name
        print("Detected:", category_name)

        global last_gestures
        last_gestures.append((category_name, timestamp_ms))

        # Keep only the last few gestures to prevent memory issues
        if len(last_gestures) > 10:
            last_gestures.pop(0)

        detect_transitions_and_features()
    else:
        print("No gestures recognized")


# Custom Gesture

from main5 import (
    cam_height,
    cam_width,
    frame,
    x_pos_buffer,
    y_pos_buffer,
    POSITION_SMOOTHING_WINDOW,
    FRAME_MARGIN_X,
    FRAME_MARGIN_Y,
    screen_height,
    screen_width,
    CLICK_COOLDOWN,
    FPS,
)


def is_palm_facing(hand_landmarks, is_right_hand):
    thumb_base = hand_landmarks.landmark[1]
    pinky_base = hand_landmarks.landmark[17]
    wrist = hand_landmarks.landmark[0]
    middle_base = hand_landmarks.landmark[9]

    # Additional check for hand orientation
    # For right hand: thumb should be left of pinky AND hand direction should be rightward
    # For left hand: thumb should be right of pinky AND hand direction should be leftward
    if is_right_hand:
        return thumb_base.x < pinky_base.x
    return thumb_base.x > pinky_base.x


def is_finger_pointing(hand_landmarks):
    """Check if index finger is pointing up while other fingers are closed"""
    # Get coordinates of index finger landmarks from base to tip
    index_mcp = hand_landmarks.landmark[5]  # Base
    index_pip = hand_landmarks.landmark[6]  # First joint
    index_dip = hand_landmarks.landmark[7]  # Second joint
    index_tip = hand_landmarks.landmark[8]  # Tip

    # Get y-coordinates for other fingers
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y

    # Minimum height difference between segments (in normalized coordinates)
    MIN_HEIGHT_DIFF = (
        0.02  # Adjust this value to control how "straight" the finger needs to be
    )

    # Check if each segment is higher than the previous one by the minimum difference
    is_ascending = (
        (index_tip.y + MIN_HEIGHT_DIFF < index_dip.y)
        and (index_dip.y + MIN_HEIGHT_DIFF < index_pip.y)
        and (index_pip.y + MIN_HEIGHT_DIFF < index_mcp.y)
    )

    # Check if index is extended upward
    index_extended = index_tip.y < index_mcp.y

    # Check if other fingers are curled
    others_curled = all(
        [
            middle_tip > index_pip.y - 0.1,
            ring_tip > index_pip.y - 0.1,
            pinky_tip > index_pip.y - 0.1,
        ]
    )

    return is_ascending and index_extended and others_curled


def is_index_tap(hand_landmarks, prev_y):
    """Detect if index finger is touching thumb"""
    # Get thumb and index finger tip positions
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # Calculate distance between thumb and index finger tips
    distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2
        + (thumb_tip.y - index_tip.y) ** 2
        + (thumb_tip.z - index_tip.z) ** 2
    )

    # Calculate z average for scaling
    z_values = [landmark.z for landmark in hand_landmarks.landmark]
    current_z_avg = sum(z_values) / len(z_values)

    # Scale threshold based on z distance with adjusted ranges
    BASE_TOUCH_THRESHOLD = 0.07

    # Adjust scaling to be more reasonable at close range
    # Map z_avg from [-0.15, -0.01] to [1.5, 0.8]
    if current_z_avg > -0.01:  # Very close to camera
        scale_factor = 0.8
    else:
        scale_factor = map_range(
            current_z_avg,
            -0.15,  # furthest expected z
            -0.01,  # closest expected z
            1.5,  # scale up threshold when far
            0.8,  # scale down threshold when close
        )

    scaled_threshold = BASE_TOUCH_THRESHOLD * scale_factor
    scaled_threshold = max(0.05, min(0.1, scaled_threshold))  # Tighter clamp range

    # Draw line between thumb and index
    thumb_pixel = (int(thumb_tip.x * cam_width), int(thumb_tip.y * cam_height))
    index_pixel = (int(index_tip.x * cam_width), int(index_tip.y * cam_height))

    # Draw line in green if touching, red if not
    color = (0, 255, 0) if distance < scaled_threshold else (0, 0, 255)
    cv2.line(frame, thumb_pixel, index_pixel, color, 2)

    # Draw distance, threshold, and z values
    mid_point = (
        (thumb_pixel[0] + index_pixel[0]) // 2,
        (thumb_pixel[1] + index_pixel[1]) // 2,
    )
    cv2.putText(
        frame,
        f"D: {distance:.2f} T: {scaled_threshold:.2f}",
        mid_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

    # Add z value display slightly below
    z_point = (mid_point[0], mid_point[1] + 20)
    cv2.putText(
        frame,
        f"Z: {current_z_avg:.2f}",
        z_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

    return distance < scaled_threshold


def map_range(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def get_smoothed_position(new_x, new_y):
    """Smooth the x,y coordinates using a simpler, faster moving average"""
    x_pos_buffer.append(new_x)
    y_pos_buffer.append(new_y)

    if len(x_pos_buffer) > POSITION_SMOOTHING_WINDOW:
        x_pos_buffer.pop(0)
        y_pos_buffer.pop(0)

    # Simple average instead of weighted for faster processing
    smoothed_x = sum(x_pos_buffer) / len(x_pos_buffer)
    smoothed_y = sum(y_pos_buffer) / len(y_pos_buffer)

    return int(smoothed_x), int(smoothed_y)


def map_coordinates_to_screen(x, y, frame_width, frame_height):
    """Map coordinates from the usable frame area to full screen with better edge handling"""
    # Calculate the usable frame area
    margin_x = frame_width * FRAME_MARGIN_X
    margin_y = frame_height * FRAME_MARGIN_Y

    # Calculate normalized position within the usable area
    frame_x = x * frame_width
    frame_y = y * frame_height

    usable_width = frame_width * (1 - 2 * FRAME_MARGIN_X)
    usable_height = frame_height * (1 - 2 * FRAME_MARGIN_Y)

    # Map to screen coordinates with padding for better edge detection
    screen_x = map_range(
        frame_x,
        margin_x,  # from: left margin
        frame_width - margin_x,  # to: right margin
        0,  # maps to: left edge of screen
        screen_width,  # maps to: right edge of screen
    )

    screen_y = map_range(
        frame_y,
        margin_y,  # from: top margin
        frame_height - margin_y,  # to: bottom margin
        0,  # maps to: top edge of screen
        screen_height,  # maps to: bottom edge of screen
    )

    # Clamp values to screen boundaries
    screen_x = max(0, min(screen_x, screen_width))
    screen_y = max(0, min(screen_y, screen_height))

    return int(screen_x), int(screen_y)


def is_pinky_tap(hand_landmarks, prev_y):
    """Detect if pinky has moved down significantly"""
    if prev_y is None:
        return False

    pinky_tip = hand_landmarks.landmark[20].y
    movement = pinky_tip - prev_y
    return movement > CLICK_THRESHOLD


# Replace pinch-related variables with finger-related ones
last_index_y = None

frametime = 1.0 / FPS
prev_time = time.time()


def is_facing_forward(face_landmarks, frame):
    """Check if the face is looking forward enough for gestures"""
    frame_h, frame_w, _ = frame.shape
    face_3d = []
    face_2d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            if idx == 1:
                nose_2d = (lm.x * frame_w, lm.y * frame_h)
                nose_3d = (lm.x * frame_w, lm.y * frame_h, lm.z * 3000)

            x, y = int(lm.x * frame_w), int(lm.y * frame_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * frame_w
    cam_matrix = np.array(
        [[focal_length, 0, frame_h / 2], [0, focal_length, frame_w / 2], [0, 0, 1]]
    )

    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    if not success:
        return False

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    if y <= -3:
        return 0
    elif y >= 2.8:
        return 0
    elif x <= 0.4:
        return 0
    elif x >= 4.4:
        return 0
    else:
        return 1
