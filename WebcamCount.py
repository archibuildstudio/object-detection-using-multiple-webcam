import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision import BoxAnnotator, Detections

print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "No GPU")

ZONE_POLYGON = np.array([
    [0, 0],
    [1280, 0],
    [1280, 720],
    [0, 720]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live Webcam Detection")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int,
        help="Set webcam resolution (width height)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")
    model.to('cuda') #uses the GPU usage

    # Create annotators
    box_annotator = BoxAnnotator()

    # Define the PolygonZone
    zone = sv.PolygonZone(polygon=ZONE_POLYGON)

    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        text_thickness=2,
        text_scale=0.8
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame. Check webcam.")
            break

        # Run inference
        results = model(frame)[0]

        # Convert to supervision.Detections
        detections = Detections.from_ultralytics(results)

        # Add class name labels to detections
        class_names = model.model.names
        detections.data["class_name"] = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Filter detections in the zone
        detections_in_zone = detections[zone.trigger(detections=detections)]

        # Annotate boxes only
        frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections_in_zone
        )

        # Manually draw class name text
        for box, label in zip(detections_in_zone.xyxy, detections_in_zone.data["class_name"]):
            x1, y1, _, _ = map(int, box)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw zone polygon
        frame = zone_annotator.annotate(scene=frame)

        # Count and show number of objects in the zone
        object_count = len(detections_in_zone)
        cv2.putText(frame, f"Objects: {object_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show result
        cv2.imshow("YOLOv8 Live", frame)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
