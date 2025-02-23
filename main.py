import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from mac_actions import (
    volume_up,
    volume_down,
    switch_desktop_left,
    switch_desktop_right,
    open_app_fullscreen
)

# Initialize MediaPipe Hand detection
ENABLE_HEAD_TRACKING = True
SHOW_CAMERA_FEED = True  # Add this flag to toggle camera feed display
CAMERA_DEVICE_ID = 0

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,  # Lowered further for better long-distance detection
    min_tracking_confidence=0.2,  # Lowered further for better tracking
    model_complexity=1,  # Increased from 0 for better accuracy at distance
)
mp_draw = mp.solutions.drawing_utils

# Add smoothing window size constant
SMOOTHING_WINDOW = 5
z_avg_buffer = []

# Add these constants near the top with other constants
POSITION_SMOOTHING_WINDOW = 3  # Reduced from 4 for less lag
x_pos_buffer = []
y_pos_buffer = []
MIN_MOVEMENT_THRESHOLD = 1  # Reduced from 2 for more responsive movement

FPS = 30.0

# Initialize webcam
cap = cv2.VideoCapture(CAMERA_DEVICE_ID)

# Set webcam resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Get webcam dimensions (actual dimensions after setting)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera resolution: {cam_width}x{cam_height}")  # Add this to verify the resolution

# Disable pyautogui's failsafe
pyautogui.FAILSAFE = False

# Add constants near the top with other constants
last_mouse_x = None
last_mouse_y = None

# Adjust the constants for asymmetric margins
FRAME_MARGIN_X = 0.1  # 10% margin on left/right
FRAME_MARGIN_Y = 0.15  # 15% margin on top/bottom - more vertical space for detection

# Add these constants near the top with other constants
CLICK_COOLDOWN = 0.2  # Adjusted for thumb-index clicking
last_click_time = 0
last_pinky_y = None

# Add these constants near the top with other constants
FINGER_CONFIRMATION_FRAMES = 2  # Reduced from 3 to detect pointing faster
finger_counter = 0  # Track consecutive pointing frames
was_pointing = False  # Track previous pointing state

# Add these constants near the top with other constants
POINTING_GRACE_PERIOD = 0.3  # Increased from 0.2 to handle fast movements better
last_pointing_time = 0  # Track when we last saw pointing gesture


# Update these constants for more responsive swipes
SWIPE_THRESHOLD = 0.04  # Even lower threshold for quicker detection
SWIPE_COOLDOWN = 0.1  # Shorter cooldown
SWIPE_MIN_DURATION = 0.0  # Remove minimum duration requirement
SWIPE_MAX_DURATION = 0.4  # Shorter maximum duration for faster response
SWIPE_VERTICAL_TOLERANCE = 3.0  # Even more vertical tolerance

# Add these variables before the main loop
last_hand_positions = []
last_swipe_time = 0


# Add these imports at the top with other imports
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Add after MediaPipe Hand initialization
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Hand detection
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model Path + Options
model_path = "./gesture_recognizer.task"
base_options = BaseOptions(model_asset_path=model_path)


# Get screen size
screen_width, screen_height = pyautogui.size()

# Get webcam dimensions
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Disable pyautogui's failsafe
pyautogui.FAILSAFE = False


# ------------------------------------------------------------------------------------------------------------------------


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
    elif last_gestures[-1][0] == "ILoveYou":
        open_app_fullscreen("Safari")
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


def gesture_result_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
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
    MIN_HEIGHT_DIFF = 0.02  # Adjust this value to control how "straight" the finger needs to be

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
        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2 + (thumb_tip.z - index_tip.z) ** 2
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
    cam_matrix = np.array([[focal_length, 0, frame_h / 2], [0, focal_length, frame_w / 2], [0, 0, 1]])

    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

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


def detect_hand_swipe(hand_landmarks, current_time, is_right_hand):
    """
    Detect horizontal hand swipes with optimized responsiveness.
    """
    global last_hand_positions, last_swipe_time

    # Skip if too soon after last swipe
    if current_time - last_swipe_time < SWIPE_COOLDOWN:
        return

    # Only track fingertips for faster processing
    fingertip_indices = [4, 8, 12, 16, 20]  # just fingertips

    current_points = [(hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y) for i in fingertip_indices]

    # Add current position
    last_hand_positions.append((current_points, current_time, is_right_hand))

    # Keep only very recent positions
    while last_hand_positions and current_time - last_hand_positions[0][1] > SWIPE_MAX_DURATION:
        last_hand_positions.pop(0)

    # Need at least 2 positions
    if len(last_hand_positions) < 2:
        return

    # Get start and end positions
    start_points, start_time, start_hand = last_hand_positions[0]
    end_points, end_time, end_hand = last_hand_positions[-1]

    # Basic duration check
    duration = end_time - start_time
    if duration > SWIPE_MAX_DURATION:  # Only check maximum duration
        return

    # Calculate movements (simplified)
    movements = [end_point[0] - start_point[0] for start_point, end_point in zip(start_points, end_points)]

    # Get average horizontal movement
    avg_movement = sum(movements) / len(movements)

    # Quick threshold check
    if abs(avg_movement) < SWIPE_THRESHOLD:
        return

    # Simple direction check
    if (is_right_hand and avg_movement > 0) or (not is_right_hand and avg_movement < 0):
        return

    # Valid swipe detected
    last_swipe_time = current_time
    last_hand_positions.clear()

    if avg_movement > 0:  # Moving right (must be left hand)
        switch_desktop_left()
        print("Left hand swipe right detected - moving left")
    else:  # Moving left (must be right hand)
        switch_desktop_right()
        print("Right hand swipe left detected - moving right")


# ------------------------------------------------------------------------------------------------------------------------


# Gesture Options

gesture_options = GestureRecognizerOptions(
    base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_result_callback,
)

# Other version
with GestureRecognizer.create_from_options(gesture_options) as recognizer:
    while True:
        curr_time = time.time()
        elapsed = curr_time - prev_time
        time.sleep(max(0, frametime - elapsed))

        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Enhance preprocessing for better long-distance detection
        # 1. Adjust contrast and brightness more aggressively
        alpha = 1.5  # Increased contrast (was 1.2)
        beta = -20  # Adjusted brightness (was -30)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # 2. Apply sharpening to enhance edge detection
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        frame = cv2.filter2D(frame, -1, kernel)

        # 3. Increase image size for better detection
        scale_factor = 1.5
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # 4. Apply minimal blur to reduce noise while keeping details
        frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Reduced kernel size

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(frame_rgb)

        # Process face landmarks
        face_results = face_mesh.process(frame_rgb)
        facing_forward = False

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                facing_forward = is_facing_forward(face_landmarks, frame)
                # Draw face mesh landmarks for visual feedback
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

        # Scale back to original size for display and coordinate mapping
        frame = cv2.resize(frame, (cam_width, cam_height))

        facing_forward = facing_forward and ENABLE_HEAD_TRACKING

        # Only process hand gestures if facing forward
        if results.multi_hand_landmarks and facing_forward:  # Added facing_forward check
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get handedness confidence and label
                handedness_score = handedness.classification[0].score
                is_right_hand = handedness.classification[0].label == 'Right'

                # Skip only if confidence is too low
                if handedness_score < 0.8:  # Removed the right hand check
                    continue

                # Calculate z average with smoothing
                z_values = [landmark.z for landmark in hand_landmarks.landmark]
                current_z_avg = sum(z_values) / len(z_values)

                # Add to buffer and maintain window size
                z_avg_buffer.append(current_z_avg)
                if len(z_avg_buffer) > SMOOTHING_WINDOW:
                    z_avg_buffer.pop(0)

                # Calculate smoothed z average
                z_avg = sum(z_avg_buffer) / len(z_avg_buffer)
                print(f"Smoothed z_avg: {z_avg:.3f}")

                # First check for pointing/cursor movement if it's the right hand
                if is_palm_facing(hand_landmarks, is_right_hand) and is_right_hand:
                    # Media Pipe Model Gesture Checking
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    current_time_ms = int(time.time() * 1000)
                    detected_gestures = recognizer.recognize_async(mp_image, current_time_ms)

                    # Check for pointing gesture
                    is_pointing = is_finger_pointing(hand_landmarks)
                    current_time = time.time()

                    if is_pointing:
                        finger_counter += 1
                        last_pointing_time = current_time
                    else:
                        finger_counter = 0

                    # Process if either pointing or within grace period
                    if finger_counter >= FINGER_CONFIRMATION_FRAMES or (
                        was_pointing and current_time - last_pointing_time < POINTING_GRACE_PERIOD
                    ):

                        was_pointing = True
                        # Get index finger positions - MCP for movement, tip for clicking
                        index_mcp = hand_landmarks.landmark[5]  # Base of index finger
                        index_tip = hand_landmarks.landmark[8]  # Tip for click detection

                        if current_time - last_click_time > CLICK_COOLDOWN and is_index_tap(hand_landmarks, None):
                            pyautogui.click(_pause=False)
                            last_click_time = current_time

                        # Only update cursor position if actually pointing (not in grace period)
                        if finger_counter >= FINGER_CONFIRMATION_FRAMES:
                            raw_x, raw_y = map_coordinates_to_screen(index_mcp.x, index_mcp.y, cam_width, cam_height)

                            # Get smoothed position
                            mouse_x, mouse_y = get_smoothed_position(raw_x, raw_y)

                            # Simplified movement logic
                            if last_mouse_x is None or last_mouse_y is None:
                                pyautogui.moveTo(mouse_x, mouse_y, _pause=False)
                                last_mouse_x = mouse_x
                                last_mouse_y = mouse_y
                            else:
                                # Calculate the movement delta
                                delta_x = mouse_x - last_mouse_x
                                delta_y = mouse_y - last_mouse_y

                                movement_delta = np.sqrt(delta_x**2 + delta_y**2)
                                if movement_delta > MIN_MOVEMENT_THRESHOLD:
                                    pyautogui.moveTo(mouse_x, mouse_y, _pause=False)
                                    last_mouse_x = mouse_x
                                    last_mouse_y = mouse_y

                            # Draw circles at both tracking points
                            cv2.circle(
                                frame,
                                (int(index_mcp.x * cam_width), int(index_mcp.y * cam_height)),
                                10,
                                (0, 255, 0),
                                -1,
                            )  # Green circle for movement point
                            cv2.circle(
                                frame, (int(index_tip.x * cam_width), int(index_tip.y * cam_height)), 8, (255, 0, 0), -1
                            )  # Red circle for click detection point
                        continue  # Skip swipe detection if we're pointing

                # Only check for swipes if we're not pointing
                detect_hand_swipe(hand_landmarks, time.time(), is_right_hand)

        # Add text to show if facing forward and control status
        status_text = "Controls Active" if facing_forward else "Face Forward to Enable Controls"
        status_color = (0, 255, 0) if facing_forward else (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Display the frame if SHOW_CAMERA_FEED is True
        try:
            if SHOW_CAMERA_FEED:
                cv2.imshow("Hand Tracking", frame)
            else:
                # Try to close the window if it exists
                try:
                    cv2.destroyWindow("Hand Tracking")
                except:
                    pass

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except:
            pass

        prev_time = curr_time

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
