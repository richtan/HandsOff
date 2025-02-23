import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Get webcam dimensions
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Disable pyautogui's failsafe
pyautogui.FAILSAFE = False

def is_palm_facing(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    hand_base = hand_landmarks.landmark[0]

    # Check that thumb is to left of pinky and thumb is also left of base of hand
    # to make sure palm is facing screen
    return thumb_tip.x < pinky_tip.x and thumb_tip.x < hand_base.x

def is_pinch_gesture(hand_landmarks):
    """Check if thumb and index finger are close enough"""
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    
    distance = np.sqrt(((thumb_tip - index_tip) ** 2).sum())
    return distance < 0.06  # Adjust this threshold as needed

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
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_palm_facing(hand_landmarks):
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
                    cv2.circle(frame,
                             (int(index_tip.x * cam_width), int(index_tip.y * cam_height)),
                             10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
