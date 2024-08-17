import cv2
import numpy as np

def create_mask(hsv_frame, lower_color, upper_color):
    """Create a mask for the specified color range."""
    return cv2.inRange(hsv_frame, lower_color, upper_color)

def find_largest_contour(mask):
    """Find the largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def track_object(frame, contour):
    """Draw bounding box and centroid for the tracked object."""
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            return (cx, cy)
    return None

def main():
    cap = cv2.VideoCapture(0)  # 0 is used for webcam input (i thik)

    # Define the colour range of the objects you want to track here in RGB
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = create_mask(hsv_frame, lower_blue, upper_blue)
        
        # Apply some morphological operations to reduce noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contour = find_largest_contour(mask)
        centroid = track_object(frame, contour)

        if centroid:
            print(f"Object centroid: {centroid}")

        cv2.imshow('Color Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Yes Harrison I did code this