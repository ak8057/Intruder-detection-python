import cv2
import numpy as np
from datetime import datetime
import os

class CCTVAnalyzer:
    def __init__(self, video_source):
        self.video = cv2.VideoCapture(video_source)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
        self.min_area = 500  # Minimum area to consider as motion
        self.accident_threshold = 5000  # Threshold for accident detection (adjust as needed)
        
        # Initialize object detector
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        
        # Create a directory for storing screenshots
        self.screenshot_dir = "intruder_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def analyze_frame(self, frame):
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Threshold the mask
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                self.detect_intruder(frame, contour)
                self.detect_accident(frame, contour, area)
            
        self.detect_unauthorized_objects(frame)

    def detect_intruder(self, frame, contour):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Intruder Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.generate_alarm("Intruder")
        self.capture_screenshot(frame, "intruder")

    def detect_accident(self, frame, contour, area):
        if area > self.accident_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Accident Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.generate_alarm("Accident")

    def detect_unauthorized_objects(self, frame):
        # Apply object detection
        mask = self.object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Unauthorized Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                self.generate_alarm("Unauthorized Object")

    def generate_alarm(self, event_type):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"ALARM: {event_type} detected at {timestamp}")
        # Add code here to trigger additional alarm mechanisms (e.g., send email, push notification, etc.)

    def capture_screenshot(self, frame, event_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/{event_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            
            self.analyze_frame(frame)
            
            cv2.imshow("CCTV Analysis", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source = 0  # Use 0 for webcam or provide path to video file
    # video_source = "thief_video.mp4"
    analyzer = CCTVAnalyzer(video_source)
    analyzer.run()