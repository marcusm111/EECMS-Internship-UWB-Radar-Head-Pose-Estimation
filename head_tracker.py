import cv2
import numpy as np
import mediapipe as mp
import math

# Input and output video paths
input_video = 0
output_video = "head_pose_output_mediapipe.mp4"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Open input video
cap = cv2.VideoCapture(input_video)

# Get original video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cropped_width = orig_width // 2  # Use only the right half

# Create VideoWriter object with new width
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (cropped_width, height))

# Face mesh key points indices for pose estimation
NOSE_TIP = 4
CHIN = 152
LEFT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
FOREHEAD = 10  # Additional point to improve pose estimation

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, 45.0, -20.0),          # Chin
    (-20.0, -20.0, -15.0),       # Left eye left corner
    (20.0, -20.0, -15.0),        # Right eye right corner
    (-15.0, 15.0, -10.0),        # Left mouth corner
    (15.0, 15.0, -10.0),         # Right mouth corner
    (0.0, -35.0, -10.0)          # Forehead
], dtype=np.float64)

frame_idx = 0
while cap.isOpened() and frame_idx < 100:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the right half of the frame (assumes a perfect vertical split)
    frame = frame[:, orig_width // 2:orig_width]
    # Update the width to match the cropped frame
    width = frame.shape[1]

    # Update progress
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processing frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and get facial landmarks
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the landmarks as a list
            landmarks = face_landmarks.landmark
            
            # Extract specific points from MediaPipe
            image_points = np.array([
                (int(landmarks[NOSE_TIP].x * width), int(landmarks[NOSE_TIP].y * height)),
                (int(landmarks[CHIN].x * width), int(landmarks[CHIN].y * height)),
                (int(landmarks[LEFT_EYE_LEFT_CORNER].x * width), int(landmarks[LEFT_EYE_LEFT_CORNER].y * height)),
                (int(landmarks[RIGHT_EYE_RIGHT_CORNER].x * width), int(landmarks[RIGHT_EYE_RIGHT_CORNER].y * height)),
                (int(landmarks[LEFT_MOUTH_CORNER].x * width), int(landmarks[LEFT_MOUTH_CORNER].y * height)),
                (int(landmarks[RIGHT_MOUTH_CORNER].x * width), int(landmarks[RIGHT_MOUTH_CORNER].y * height)),
                (int(landmarks[FOREHEAD].x * width), int(landmarks[FOREHEAD].y * height))
            ], dtype="double")
            
            # Camera parameters (updated for cropped frame)
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))
            
            # Find pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Calculate angles directly from rotation matrix
                sin_pitch = -rotation_matrix[2, 0]
                pitch_deg = math.degrees(math.asin(sin_pitch))
                
                if abs(sin_pitch) > 0.99:
                    yaw_deg = math.degrees(math.atan2(rotation_matrix[0, 1], rotation_matrix[1, 1]))
                else:
                    yaw_deg = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
                
                if abs(sin_pitch) > 0.99:
                    roll_deg = 0.0
                else:
                    roll_deg = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
                
                # Adjust signs for intuitive directions
                yaw_deg = -yaw_deg
                roll_deg = -roll_deg
                
                # Create text strings with the calculated angles
                pitch_text = f"Pitch: {pitch_deg:.2f}"
                yaw_text = f"Yaw: {yaw_deg:.2f}"
                roll_text = f"Roll: {roll_deg:.2f}"
                
                # Determine movement text based on thresholds
                movement_text = ""
                if abs(pitch_deg) > 15:
                    movement_text += "Head " + ("UP" if pitch_deg > 0 else "DOWN") + " "
                if abs(yaw_deg) > 15:
                    movement_text += "Looking " + ("RIGHT" if yaw_deg > 0 else "LEFT") + " "
                if abs(roll_deg) > 15:
                    movement_text += "Tilted " + ("RIGHT" if roll_deg > 0 else "LEFT")
                
                # Draw axes to visualize head pose
                axis_length = 70
                nose = (int(image_points[0][0]), int(image_points[0][1]))
                
                # Define 3D axes points
                axis_points = np.array([
                    [axis_length, 0, 0],      # X-axis (red)
                    [0, axis_length, 0],      # Y-axis (green)
                    [0, 0, axis_length]       # Z-axis (blue)
                ], dtype=np.float64)
                
                # Project the 3D axis points to the image plane
                projected_points, _ = cv2.projectPoints(
                    axis_points, 
                    rotation_vector, 
                    translation_vector, 
                    camera_matrix, 
                    dist_coeffs
                )
                
                x_end = (int(nose[0] + projected_points[0][0][0]), int(nose[1] + projected_points[0][0][1]))
                y_end = (int(nose[0] + projected_points[1][0][0]), int(nose[1] + projected_points[1][0][1]))
                z_end = (int(nose[0] + projected_points[2][0][0]), int(nose[1] + projected_points[2][0][1]))
                
                cv2.line(frame, nose, x_end, (0, 0, 255), 2)    # X-axis (red)
                cv2.line(frame, nose, y_end, (0, 255, 0), 2)      # Y-axis (green)
                cv2.line(frame, nose, z_end, (255, 0, 0), 2)      # Z-axis (blue)
                
                cv2.putText(frame, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                angle_explanation = "PITCH: Up(+)/down(-) | YAW: Right(+)/left(-) | ROLL: Tilt right(+)/left(-)"
                cv2.putText(frame, angle_explanation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw the face mesh for visualization (optional)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
            )
            
            # Calculate bounding box for text placement
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            cv2.putText(frame, pitch_text, (x_min, y_min - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, yaw_text, (x_min, y_min - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, roll_text, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if movement_text:
                cv2.putText(frame, movement_text, (x_min, y_min - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Write the processed frame to output
    out.write(frame)
    
    # Optionally display the frame
    cv2.imshow("Head Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
face_mesh.close()

print("Video processing complete.")
