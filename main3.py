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
#ADDD THISSS
# Initalize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

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


while True:
    # Set to a direction that wouldn't trigger a response
    direction = "DOWN"

    success, frame = cap.read()

    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    face_results = face_mesh.process(frame_rgb)

    frame_h, frame_w, frame_c = frame_rgb.shape

    face_3d = []
    face_2d = []

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * frame_w, lm.y * frame_h)
                        nose_3d = (lm.x * frame_w, lm.y * frame_h, lm.z * 3000)
                    
                    x, y = int(lm.x * frame_w), int(lm.y * frame_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * frame_w

            cam_matrix = np.array([ [focal_length, 0, frame_h / 2],
                                    [0, focal_length, frame_w / 2],
                                    [0, 0, 1]])
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            face_success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y <= -4:
                direction = "Left"
            elif y >= 4:
                direction = "Right"
            elif x <= 0:
                direction = "Down"
            elif x >= 8:
                direction = "Up"
            else:
                direction = "Forward"

            nose_3d_proj, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1])) 
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10)) 
            cv2.line(frame, p1, p2, (255, 0, 0), 3)
            mp_drawing.draw_landmarks(image = frame, landmark_list = face_landmarks, connections = mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)


    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            is_right_hand = handedness.classification[0].label == 'Right'

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
