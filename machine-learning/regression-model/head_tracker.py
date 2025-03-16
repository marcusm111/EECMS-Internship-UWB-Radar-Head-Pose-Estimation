import sys
import torchvision

# Add the cloned 6DRepNet360 repository to the Python path
sys.path.append(r"C:\Users\josep\Repos\EECMS-Internship2024-WiFi-Analysis\machine-learning\sixdrepnet360")  # or use the full/relative path if needed

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Import RetinaFace for face detection
from retinaface import RetinaFace

from sixdrepnet360 import SixDRepNet360  # assuming your repo is in PYTHONPATH
from sixdrepnet360 import utils  # Make sure utils is in your PYTHONPATH as well

class SixDRepNet360Wrapper(SixDRepNet360):
    def __init__(self, block, layers, fc_layers=1):
        super(SixDRepNet360Wrapper, self).__init__(block, layers, fc_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # Define the same preprocessing as in the repository's main script:
        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, frame):
        """
        Accepts a BGR image (from OpenCV), detects the face using RetinaFace,
        crops the face region, processes it, and returns the predicted pitch, yaw, and roll angles (in degrees).
        """
        # Use RetinaFace for face detection.
        try:
            detections = RetinaFace.detect_faces(frame)
            if isinstance(detections, dict) and len(detections) > 0:
                # Choose the face with the highest confidence score.
                best_face = max(detections.values(), key=lambda x: x["score"])
                bbox = best_face["facial_area"]  # Expected format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                # Ensure the bounding box is within the image bounds.
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_frame = frame[y1:y2, x1:x2]
            else:
                # Fallback: if no face is detected, use the full frame.
                face_frame = frame
        except Exception as e:
            print("Error in face detection, using full image:", e)
            face_frame = frame

        # Convert BGR to RGB and then to a PIL image
        frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Preprocess the image
        input_tensor = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        # Run inference
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            rot_matrix = self.forward(input_tensor)

        # Convert rotation matrix to Euler angles using the utils function.
        # Note: The repository computes:
        #   euler = utils.compute_euler_angles_from_rotation_matrices(rot_matrix) * 180/np.pi
        euler_angles = utils.compute_euler_angles_from_rotation_matrices(rot_matrix) * 180/np.pi

        # The repository code assumes:
        #   pitch = euler_angles[0, 0], yaw = euler_angles[0, 1], roll = euler_angles[0, 2]
        pitch = euler_angles[0, 0].item()
        yaw   = euler_angles[0, 1].item()
        roll  = euler_angles[0, 2].item()
        return pitch, yaw, roll

# Input and output video paths
input_video = "output0.mp4"
output_video = "head_pose_output_6DRepNet360.mp4"

# Open input video
cap = cv2.VideoCapture(input_video)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create VideoWriter object (output video has same resolution)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Initialize the 6DRepNet360 model wrapper.
# Here we use torchvision.models.resnet.Bottleneck and the ResNet-50 style layer configuration.
model = SixDRepNet360Wrapper(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)

saved_state_dict = torch.load("weights/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth")
model.load_state_dict(saved_state_dict)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"Processing frame {frame_idx}/{frame_count}")

    # Predict head pose using the model's predict method.
    pitch, yaw, roll = model.predict(frame)

    # Overlay the head pose angles on the frame
    text = f"Pitch: {pitch:.2f}  Yaw: {yaw:.2f}  Roll: {roll:.2f}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the pose cube visualization or axes if desired.
    # frame = utils.plot_pose_cube(frame, yaw, pitch, roll)
    frame = utils.draw_axis(frame, yaw, pitch, roll)

    # Write the processed frame to output video
    out.write(frame)

    # Display the frame (press ESC to exit)
    cv2.imshow("Head Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")
