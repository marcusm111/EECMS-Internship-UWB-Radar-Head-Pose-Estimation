import cv2
from sixdrepnet import SixDRepNet

# Input and output video paths
input_video = "c1p1r11.mp4"
output_video = "head_pose_output_sixdrepnet.mp4"

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

# Initialize sixdrepnet model (weights are downloaded automatically)
model = SixDRepNet()

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"Processing frame {frame_idx}/{frame_count}")

    # Predict head pose using sixdrepnet
    # Returns pitch, yaw, roll (in degrees)
    pitch, yaw, roll = model.predict(frame)

    # Overlay the head pose angles on the frame
    text = f"Pitch: {pitch:.2f}  Yaw: {yaw:.2f}  Roll: {roll:.2f}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Optionally, you can draw axes using:
    # model.draw_axis(frame, yaw, pitch, roll)

    # Write the processed frame to output
    out.write(frame)

    # Display the result (press ESC to exit)
    cv2.imshow("Head Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")
