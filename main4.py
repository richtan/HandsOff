import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from mac_actions import volume_up, volume_down
from custom_gestures import (
    gesture_result_callback,
    is_palm_facing,
    is_finger_pointing,
    is_index_tap,
    map_range,
    get_smoothed_position,
    map_coordinates_to_screen,
    is_pinky_tap,
    is_facing_forward,
)

# Initialize MediaPipe Hand detection
ENABLE_HEAD_TRACKING = True
SHOW_CAMERA_FEED = True  # Add this flag to toggle camera feed display

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
cap = cv2.VideoCapture(0)

# Set webcam resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Get webcam dimensions (actual dimensions after setting)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(
    f"Camera resolution: {cam_width}x{cam_height}"
)  # Add this to verify the resolution

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


# Replace pinch-related variables with finger-related ones
last_index_y = None

frametime = 1.0 / FPS
prev_time = time.time()

# Add these imports at the top with other imports
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Add after MediaPipe Hand initialization
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Initialize MediaPipe Hand detection
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model Path + Options
model_path = "./gesture_recognizer.task"
base_options = BaseOptions(model_asset_path=model_path)
gesture_options = GestureRecognizerOptions(
    base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_result_callback,
)

# Initialize webcam
cap = cv2.VideoCapture(1)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Get webcam dimensions
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Disable pyautogui's failsafe
pyautogui.FAILSAFE = False


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
        if (
            results.multi_hand_landmarks and facing_forward
        ):  # Added facing_forward check
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get handedness confidence and label
                handedness_score = handedness.classification[0].score
                is_right_hand = handedness.classification[0].label == "Right"

                # Skip if confidence is too low or not right hand
                if (
                    handedness_score < 0.8 or not is_right_hand
                ):  # Added confidence threshold
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

                # Perform gesture recognition on the processed image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                current_time_ms = int(time.time() * 1000)

                detected_gestures = recognizer.recognize_async(
                    mp_image, current_time_ms
                )

                if is_palm_facing(hand_landmarks, is_right_hand):
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
                        was_pointing
                        and current_time - last_pointing_time < POINTING_GRACE_PERIOD
                    ):

                        was_pointing = True
                        # Get index finger positions - MCP for movement, tip for clicking
                        index_mcp = hand_landmarks.landmark[5]  # Base of index finger
                        index_tip = hand_landmarks.landmark[
                            8
                        ]  # Tip for click detection

                        if (
                            current_time - last_click_time > CLICK_COOLDOWN
                            and is_index_tap(
                                hand_landmarks, None, cam_height, cam_width, frame
                            )
                        ):
                            pyautogui.click(_pause=False)
                            last_click_time = current_time

                        # Only update cursor position if actually pointing (not in grace period)
                        if finger_counter >= FINGER_CONFIRMATION_FRAMES:
                            raw_x, raw_y = map_coordinates_to_screen(
                                index_mcp.x, index_mcp.y, cam_width, cam_height
                            )

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
                                    # Move directly to the new position without scaling or duration
                                    pyautogui.moveTo(mouse_x, mouse_y, _pause=False)
                                    last_mouse_x = mouse_x
                                    last_mouse_y = mouse_y

                            # Draw circles at both tracking points
                            cv2.circle(
                                frame,
                                (
                                    int(index_mcp.x * cam_width),
                                    int(index_mcp.y * cam_height),
                                ),
                                10,
                                (0, 255, 0),
                                -1,
                            )  # Green circle for movement point
                            cv2.circle(
                                frame,
                                (
                                    int(index_tip.x * cam_width),
                                    int(index_tip.y * cam_height),
                                ),
                                8,
                                (255, 0, 0),
                                -1,
                            )  # Red circle for click detection point

                    elif was_pointing:  # Only reset if we're past grace period
                        if current_time - last_pointing_time >= POINTING_GRACE_PERIOD:
                            was_pointing = False
                            last_mouse_x = None
                            last_mouse_y = None
                            finger_counter = 0

        # Add text to show if facing forward and control status
        status_text = (
            "Controls Active" if facing_forward else "Face Forward to Enable Controls"
        )
        status_color = (0, 255, 0) if facing_forward else (0, 0, 255)

        cv2.putText(
            frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2
        )

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
