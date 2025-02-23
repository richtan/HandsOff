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
    min_detection_confidence=0.3,  # Lowered further for better long-distance detection
    min_tracking_confidence=0.2,   # Lowered further for better tracking
    model_complexity=1  # Increased from 0 for better accuracy at distance
)
mp_draw = mp.solutions.drawing_utils

# Add smoothing window size constant
SMOOTHING_WINDOW = 5
z_avg_buffer = []

# Add these constants near the top with other constants
POSITION_SMOOTHING_WINDOW = 4  # Reduced from 12 for faster response
x_pos_buffer = []
y_pos_buffer = []
MIN_MOVEMENT_THRESHOLD = 2  # Reduced from 5 for more responsive movement

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

# Add these constants near the top with other constants
CLICK_COOLDOWN = 0.1  # Reduced from 0.5 to make clicking more responsive
last_click_time = 0
last_pinky_y = None
CLICK_THRESHOLD = 0.03  # Reduced from 0.03 to make clicks easier to trigger

# Add these constants near the top with other constants
FINGER_CONFIRMATION_FRAMES = 3  # Number of consecutive frames needed to confirm pointing
finger_counter = 0  # Track consecutive pointing frames
was_pointing = False  # Track previous pointing state

# Add these constants near the top with other constants
POINTING_GRACE_PERIOD = 0.2  # Grace period in seconds after losing pointing gesture
last_pointing_time = 0  # Track when we last saw pointing gesture

def is_palm_facing(hand_landmarks, is_right_hand):
    thumb_base = hand_landmarks.landmark[1]
    pinky_base = hand_landmarks.landmark[17]

    # Check that thumb is to left of pinky and thumb is also left of base of hand
    # to make sure palm is facing screen
    if is_right_hand:
        return thumb_base.x < pinky_base.x
    return thumb_base.x > pinky_base.x

def is_finger_pointing(hand_landmarks):
    """Check if index finger is pointing up while other fingers are closed"""
    # Get y-coordinates of finger landmarks
    index_tip = hand_landmarks.landmark[8].y
    index_pip = hand_landmarks.landmark[6].y  # Joint below tip
    index_mcp = hand_landmarks.landmark[5].y  # Base of index finger
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y
    
    # Index finger should be generally extended
    # Allow for some bending during clicking by checking against MCP instead of PIP
    index_extended = index_tip < index_mcp
    
    # Other fingers should be curled (tips below index pip)
    others_curled = all([
        middle_tip > index_pip,
        ring_tip > index_pip,
        pinky_tip > index_pip
    ])
    
    return index_extended and others_curled

def is_index_tap(hand_landmarks, prev_y):
    """Detect if index finger has moved down significantly"""
    if prev_y is None:
        return False
        
    index_tip = hand_landmarks.landmark[8].y
    index_pip = hand_landmarks.landmark[6].y  # Get PIP joint position
    
    # Check both absolute movement and relation to PIP joint
    movement = index_tip - prev_y
    tip_to_pip = index_tip - index_pip
    
    # Return true if either condition is met
    return movement > CLICK_THRESHOLD or tip_to_pip > 0.02

def map_range(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_smoothed_position(new_x, new_y):
    """Smooth the x,y coordinates using a moving average with faster response"""
    x_pos_buffer.append(new_x)
    y_pos_buffer.append(new_y)
    
    if len(x_pos_buffer) > POSITION_SMOOTHING_WINDOW:
        x_pos_buffer.pop(0)
        y_pos_buffer.pop(0)
    
    # Weight recent positions more heavily
    weights = [0.1, 0.2, 0.3, 0.4][:len(x_pos_buffer)]  # Adjust weights based on buffer size
    weight_sum = sum(weights)
    
    smoothed_x = sum(x * w for x, w in zip(x_pos_buffer, weights)) / weight_sum
    smoothed_y = sum(y * w for y, w in zip(y_pos_buffer, weights)) / weight_sum
    
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
    beta = -20   # Adjusted brightness (was -30)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # 2. Apply sharpening to enhance edge detection
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
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

    # Scale back to original size for display and coordinate mapping
    frame = cv2.resize(frame, (cam_width, cam_height))

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
                # Check for pointing gesture
                is_pointing = is_finger_pointing(hand_landmarks)
                current_time = time.time()
                
                if is_pointing:
                    finger_counter += 1
                    last_pointing_time = current_time
                else:
                    finger_counter = 0
                
                # Process if either pointing or within grace period
                if (finger_counter >= FINGER_CONFIRMATION_FRAMES or 
                    (was_pointing and current_time - last_pointing_time < POINTING_GRACE_PERIOD)):
                    
                    was_pointing = True
                    # Get index finger positions - MCP for movement, tip for clicking
                    index_mcp = hand_landmarks.landmark[5]  # Base of index finger
                    index_tip = hand_landmarks.landmark[8]  # Tip for click detection
                    
                    # Track index tip position for click detection
                    index_tip_y = index_tip.y
                    
                    if (last_index_y is not None and 
                        current_time - last_click_time > CLICK_COOLDOWN and
                        is_index_tap(hand_landmarks, last_index_y)):
                        pyautogui.click(_pause=False)
                        last_click_time = current_time
                        
                    last_index_y = index_tip_y

                    # Only update cursor position if actually pointing (not in grace period)
                    if finger_counter >= FINGER_CONFIRMATION_FRAMES:
                        raw_x, raw_y = map_coordinates_to_screen(
                            index_mcp.x, index_mcp.y,
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

                        # Draw circles at both tracking points
                        cv2.circle(frame,
                                 (int(index_mcp.x * cam_width), int(index_mcp.y * cam_height)),
                                 10, (0, 255, 0), -1)  # Green circle for movement point
                        cv2.circle(frame,
                                 (int(index_tip.x * cam_width), int(index_tip.y * cam_height)),
                                 8, (255, 0, 0), -1)  # Red circle for click detection point

                elif was_pointing:  # Only reset if we're past grace period
                    if current_time - last_pointing_time >= POINTING_GRACE_PERIOD:
                        was_pointing = False
                        last_mouse_x = None
                        last_mouse_y = None
                        last_index_y = None
                        x_pos_buffer.clear()
                        y_pos_buffer.clear()
                        finger_counter = 0

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_time = curr_time

# Release resources
cap.release()
cv2.destroyAllWindows()