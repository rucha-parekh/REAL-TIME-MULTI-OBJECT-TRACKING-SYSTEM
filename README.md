This project is a real-time object detection and tracking system tailored for flexible
application in various video analysis tasks. It leverages the YOLOv8 deep learning
model for reliable object detection and a custom Intersection over Union (IoU)-based
tracker for accurate object tracking. Supporting both video files and live webcam feeds,
the system captures key performance metrics, such as average FPS and detection
counts by object class. Designed for efficiency, it offers real-time visualization with an
option to save the processed output, making it suitable for a range of applications, from
surveillance to crowd analytics.
System Capabilities
 Object Detection: Utilizes YOLOv8 models to detect objects in each frame with
high accuracy, ensuring precise identification across diverse scenes.
 Real-Time Tracking: Employs a custom IoU-based tracking algorithm for
consistent tracking of detected objects across frames, assigning unique IDs for
uninterrupted monitoring.
 Performance Metrics: Captures and displays critical metrics, including average
FPS and detection counts per class, offering real-time insights into system
performance.
 Configurable Detection Parameters: Allows fine-tuniing with adjustable
thresholds for confidence, IoU, and maximum detections to accommodate
specific use cases.
 Multi-Source Input: Supports both video files and live camera feeds, providing
versatility for various real-world scenarios.
 Output Saving and Visualization: Includes an option to save processed videos
with overlaid bounding boxes, tracking details, and performance metrics for
post-analysis and documentation.
Key Libraries Used
 Ultralytics YOLO: Provides YOLO (You Only Look Once) object detection
models, specifically YOLOv8, enabling high-performance real-time object
detection.
 OpenCV (cv2): A powerful computer vision library used for video capture, image
processing, and visualization of detection results, including frame reading,
bounding box drawing, and real-time display.
 Logging: The built-in logging module tracks events during program execution,
aiding in debugging and monitoring with detailed logs of key events and errors.
 Threading: Manages concurrent execution, allowing object detection to run in a
separate thread so video processing and user interaction can occur
simultaneously without interruptions.
 JSON: Facilitates reading and writing data in JSON format, used here for saving
performance metrics to a file for easy sharing and further analysis.
<img width="745" height="418" alt="image" src="https://github.com/user-attachments/assets/fa53e50f-8d0a-402b-9194-feb653e25631" />
<img width="748" height="296" alt="image" src="https://github.com/user-attachments/assets/0de1361b-9f34-42ac-8dbd-a807590343f7" />
<img width="753" height="434" alt="image" src="https://github.com/user-attachments/assets/c8d244d8-62ab-49c4-8c6b-603842075756" />
