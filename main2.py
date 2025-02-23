import cv2
import mediapipe as mp
import pyautogui
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from mac_actions import volume_up, volume_down


# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Hand detection
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "./gesture_recognizer.task"
base_options = BaseOptions(model_asset_path=model_path)


last_gestures = []
# gesture_time = 0
# gesture_start_time = 0


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


options = GestureRecognizerOptions(
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


def is_palm_facing(hand_landmarks, is_right_hand):
    thumb_base = hand_landmarks.landmark[1]
    pinky_base = hand_landmarks.landmark[17]

    # Check that thumb is to left of pinky and thumb is also left of base of hand
    # to make sure palm is facing screen
    if is_right_hand:
        return thumb_base.x < pinky_base.x
    return thumb_base.x > pinky_base.x


def is_pinch_gesture(hand_landmarks):
    """Check if thumb and index finger are close enough"""
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])

    distance = np.sqrt(((thumb_tip - index_tip) ** 2).sum())
    return distance < 0.06  # Adjust this threshold as needed


with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):

                # Perform gesture recognition on the processed image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                current_time_ms = int(time.time() * 1000)

                detected_gestures = recognizer.recognize_async(
                    mp_image, current_time_ms
                )

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                is_right_hand = handedness.classification[0].label == "Right"

                if is_palm_facing(hand_landmarks, is_right_hand):
                    # Check for pinch gesture
                    if is_pinch_gesture(hand_landmarks):
                        # Get index finger tip position
                        index_tip = hand_landmarks.landmark[8]

                        # Convert the normalized coordinates to screen coordinates
                        mouse_x = int(index_tip.x * screen_width)
                        mouse_y = int(index_tip.y * screen_height)

                        # Move the mouse cursor
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0.05)

                        # Draw a circle at the cursor position on the frame
                        cv2.circle(
                            frame,
                            (
                                int(index_tip.x * cam_width),
                                int(index_tip.y * cam_height),
                            ),
                            10,
                            (0, 255, 0),
                            -1,
                        )

        # Display the frame
        cv2.imshow("Hand Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
