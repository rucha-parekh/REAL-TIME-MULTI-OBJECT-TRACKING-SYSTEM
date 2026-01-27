import cv2

# Test camera indexes 0 to 3
for i in range(4):  # Adjust range if you have more cameras
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)  # Display each camera feed for 1 second
            cv2.destroyAllWindows()
    cap.release()

cv2.destroyAllWindows()