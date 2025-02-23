import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# Add smoothing window size constant
SMOOTHING_WINDOW = 5
z_avg_buffer = []

# Add these constants near the top with other constants
POSITION_SMOOTHING_WINDOW = 12  # Increased from 8 for more smoothing
x_pos_buffer = []
y_pos_buffer = []
MIN_MOVEMENT_THRESHOLD = 5  # Base threshold for movement

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

print(f"Camera resolution: {cam_width}x{cam_height}")  # Add this to verify the resolution

# Disable pyautogui's failsafe
pyautogui.FAILSAFE = False

# Add constants near the top with other constants
last_mouse_x = None
last_mouse_y = None

# Adjust the constants for asymmetric margins
FRAME_MARGIN_X = 0.1  # 10% margin on left/right
FRAME_MARGIN_Y = 0.15  # 15% margin on top/bottom - more vertical space for detection

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

def map_range(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_smoothed_position(new_x, new_y):
    """Smooth the x,y coordinates using a moving average"""
    x_pos_buffer.append(new_x)
    y_pos_buffer.append(new_y)
    
    if len(x_pos_buffer) > POSITION_SMOOTHING_WINDOW:
        x_pos_buffer.pop(0)
        y_pos_buffer.pop(0)
    
    smoothed_x = sum(x_pos_buffer) / len(x_pos_buffer)
    smoothed_y = sum(y_pos_buffer) / len(y_pos_buffer)
    
    return int(smoothed_x), int(smoothed_y)

def map_coordinates_to_screen(x, y, frame_width, frame_height):
    """Map coordinates from the usable frame area to full screen with better edge handling"""
    # Calculate the usable frame area
    margin_x = frame_width * FRAME_MARGIN_X
    margin_y = frame_height * FRAME_MARGIN_Y
    
    # Calculate normalized position within the usable area
    frame_x = (x * frame_width)
    frame_y = (y * frame_height)
    
    usable_width = frame_width * (1 - 2 * FRAME_MARGIN_X)
    usable_height = frame_height * (1 - 2 * FRAME_MARGIN_Y)
    
    # Map to screen coordinates with padding for better edge detection
    screen_x = map_range(
        frame_x,
        margin_x,  # from: left margin
        frame_width - margin_x,  # to: right margin
        0,  # maps to: left edge of screen
        screen_width  # maps to: right edge of screen
    )
    
    screen_y = map_range(
        frame_y,
        margin_y,  # from: top margin
        frame_height - margin_y,  # to: bottom margin
        0,  # maps to: top edge of screen
        screen_height  # maps to: bottom edge of screen
    )
    
    # Clamp values to screen boundaries
    screen_x = max(0, min(screen_x, screen_width))
    screen_y = max(0, min(screen_y, screen_height))
    
    return int(screen_x), int(screen_y)

frametime = 1.0 / FPS
prev_time = time.time()

while True:
    curr_time = time.time()
    elapsed = curr_time - prev_time
    time.sleep(max(0, frametime - elapsed))
    
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
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            is_right_hand = handedness.classification[0].label == 'Right'

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

            if is_palm_facing(hand_landmarks, is_right_hand):
                # Check for pinch gesture
                if is_pinch_gesture(hand_landmarks):
                    # Get index finger tip position
                    index_tip = hand_landmarks.landmark[8]

                    # Map coordinates using the middle 80% of frame
                    raw_x, raw_y = map_coordinates_to_screen(
                        index_tip.x, index_tip.y, 
                        cam_width, cam_height
                    )

                    # Get smoothed position
                    mouse_x, mouse_y = get_smoothed_position(raw_x, raw_y)

                    # Calculate scaling factor based on z_avg
                    scaling_factor = map_range(z_avg, -0.14, -0.01, 0.1, 2.0)
                    scaling_factor = max(0.1, min(2.0, scaling_factor))

                    # Only move if this is first position or movement exceeds threshold
                    if last_mouse_x is None or last_mouse_y is None:
                        pyautogui.moveTo(mouse_x, mouse_y, _pause=False)
                    else:
                        # Calculate the movement delta
                        delta_x = mouse_x - last_mouse_x
                        delta_y = mouse_y - last_mouse_y
                        
                        # Calculate movement threshold based on z-distance
                        # When hand is closer (smaller scaling_factor), use higher threshold
                        dynamic_threshold = MIN_MOVEMENT_THRESHOLD * (1 / scaling_factor)
                        
                        movement_delta = np.sqrt(delta_x**2 + delta_y**2)
                        if movement_delta > dynamic_threshold:
                            # Apply scaling to the movement delta
                            scaled_x = last_mouse_x + (delta_x * scaling_factor)
                            scaled_y = last_mouse_y + (delta_y * scaling_factor)
                            
                            # Use longer duration when hand is closer for more stability
                            movement_duration = max(0.016, 0.016 / scaling_factor)
                            pyautogui.moveTo(scaled_x, scaled_y, 
                                           duration=movement_duration, 
                                           _pause=False)
                            last_mouse_x = scaled_x
                            last_mouse_y = scaled_y

                    # Draw a circle at the cursor position on the frame
                    cv2.circle(frame,
                             (int(index_tip.x * cam_width), int(index_tip.y * cam_height)),
                             10, (0, 255, 0), -1)
                else:
                    # Reset buffers and last position when pinch gesture ends
                    last_mouse_x = None
                    last_mouse_y = None
                    x_pos_buffer.clear()
                    y_pos_buffer.clear()

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_time = curr_time

# Release resources
cap.release()
cv2.destroyAllWindows()