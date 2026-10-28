import cv2
from ultralytics import YOLO

# Zone and counting line settings
ZONE_X1 = 1000
ZONE_Y1 = 550
ZONE_X2 = 1700
ZONE_Y2 = 750

LINE_X = 1300

# Load YOLO model
model = YOLO("yolo26n.pt")

# Open input video
cap = cv2.VideoCapture("data/line1.mp4")

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

# Read video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "data/output_zone.mp4",
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("Error: Could not create output video!")
    cap.release()
    exit()

# Counters
empty_frames = 0
total_frames = 0

# Process video frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    total_frames += 1

    # Crop the zone (rows first, then columns)
    zone = frame[ZONE_Y1:ZONE_Y2, ZONE_X1:ZONE_X2]

    # Run YOLO on the cropped zone
    results = model(
        zone,
        classes=[39],
        conf=0.25,
        device="mps",
        verbose=False
    )

    # Count frames with zero detections
    if len(results[0].boxes) == 0:
        empty_frames += 1

    # Draw each detected bottle
    for box in results[0].boxes:
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        # Convert zone coordinates to full-frame coordinates
        bx1 = int(bx1 + ZONE_X1)
        bx2 = int(bx2 + ZONE_X1)
        by1 = int(by1 + ZONE_Y1)
        by2 = int(by2 + ZONE_Y1)

        # Draw green bounding box
        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (0, 255, 0),
            2
        )

        # Display confidence above the box
        cv2.putText(
            frame,
            f"{confidence:.2f}",
            (bx1, max(by1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Draw zone rectangle in yellow
    cv2.rectangle(
        frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 255, 255),
        2
    )

    # Draw vertical counting line in red
    cv2.line(
        frame,
        (LINE_X, ZONE_Y1),
        (LINE_X, ZONE_Y2),
        (0, 0, 255),
        2
    )

    # Save processed frame
    out.write(frame)

# Release resources
cap.release()
out.release()

print(f"Total frames: {total_frames}")
print(f"Empty frames: {empty_frames}")
print("Output saved to data/output_zone.mp4")