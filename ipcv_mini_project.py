from ultralytics import YOLO
import cv2
import numpy as np
from enum import Enum
import time
import logging
from collections import defaultdict
from typing import List, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ModelType(Enum):
    YOLOV8S = 'yolov8s.pt'  # You can change this to a different YOLO model if needed

class DetectionConfig:
    """Configuration for object detection"""
    def __init__(self):
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.display_resolution = (1280, 720)
        self.enable_tracking = True

class ObjectTracker:
    """Tracks detected objects across frames"""
    def __init__(self, max_history: int = 30):
        self.tracking_history = defaultdict(lambda: [])
        self.max_history = max_history

    def update(self, detections: List[dict]):
        """Update tracking history with new detections"""
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id != -1:
                bbox = det['bbox']
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                self.tracking_history[track_id].append(center)
                if len(self.tracking_history[track_id]) > self.max_history:
                    self.tracking_history[track_id].pop(0)

class ObjectDetectionSystem:
    """Main class for object detection and tracking system"""
    def __init__(self, model_type: ModelType, config: DetectionConfig):
        self.model = YOLO(model_type.value)
        self.config = config
        self.tracker = ObjectTracker()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process a single frame and return the annotated frame and detections"""
        start_time = time.time()

        # Resize frame if needed
        if frame.shape[:2] != self.config.display_resolution[::-1]:
            frame = cv2.resize(frame, self.config.display_resolution)

        # Run YOLO detection
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            tracker="bytetrack.yaml" if self.config.enable_tracking else None
        )[0]
        
        # Process detections
        detections = []
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            class_name = results.names[cls]
            track_id = int(box.id[0]) if box.id is not None else -1
            
            detections.append({
                'bbox': bbox,
                'confidence': conf,
                'class_id': cls,
                'class_name': class_name,
                'track_id': track_id
            })

        # Update tracking
        self.tracker.update(detections)
        
        # Draw detections
        annotated_frame = self._draw_detections(frame, detections)
        
        # Log processing time
        logging.info(f"Processing time: {time.time() - start_time:.4f} seconds")
        
        return annotated_frame, detections

    def _draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame for each detection."""
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            # Draw bounding box
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            color = (0, 255, 0)  # Green color for boxes
            cv2.rectangle(frame, start_point, end_point, color, 2)

            # Draw label with confidence
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def start_detection(self, source: Union[int, str]):
        """Start detection on a video file or camera."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error("Failed to open video source")
            return
        
        source_type = "camera" if isinstance(source, int) else "video file"
        print(f"Real-time detection started on {source_type}. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, detections = self.process_frame(frame)
            cv2.imshow("Object Detection", annotated_frame)
            
            # Quit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    # Initialize system
    detection_system = ObjectDetectionSystem(ModelType.YOLOV8S, DetectionConfig())
    
    # Prompt user to choose detection source
    source_type = input("Choose detection source: (1) Camera or (2) Video File: ").strip()
    
    if source_type == "1":
        # Use default camera
        detection_system.start_detection(0)
    elif source_type == "2":
        # Ask for video file path
        video_path = input("Enter the path to the video file: ").strip()
        detection_system.start_detection(video_path)
    else:
        print("Invalid option. Please enter 1 or 2.")
