import cv2
import torch
import numpy as np
import pygame
from datetime import datetime
import os

class AdvancedCCTVAnalyzer:
    def __init__(self, video_source, alarm_path="alarm.wav"):
        self.video = cv2.VideoCapture(video_source)
        self.alarm_path = alarm_path
        self.screenshot_dir = "intruder_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Initialize pygame for audio
        pygame.init()
        pygame.mixer.music.load(self.alarm_path)

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.target_classes = ['person', 'car', 'truck', 'bus']
        self.pts = []  # Polygon points for ROI
        self.number_of_photos = 3
        self.photo_count = 0

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pts = []

    def inside_polygon(self, point, polygon):
        result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
        return result == 1

    def preprocess(self, img):
        height, width = img.shape[:2]
        ratio = height / width
        return cv2.resize(img, (640, int(640 * ratio)))

    def analyze_frame(self, frame):
        frame_detected = frame.copy()
        frame = self.preprocess(frame)

        results = self.model(frame)

        for index, row in results.pandas().xyxy[0].iterrows():
            if row['name'] in self.target_classes:
                name = str(row['name'])
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                if len(self.pts) >= 4:
                    if self.inside_polygon((center_x, center_y), np.array([self.pts])) and name == 'person':
                        self.handle_intruder(frame, frame_detected, x1, y1, x2, y2, center_x, center_y)

        if len(self.pts) >= 4:
            frame_copy = frame.copy()
            cv2.fillPoly(frame_copy, np.array([self.pts]), (0, 255, 0))
            frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)

        return frame

    def handle_intruder(self, frame, frame_detected, x1, y1, x2, y2, center_x, center_y):
        mask = np.zeros_like(frame_detected)
        points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
        mask = cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], (255, 255, 255))
        frame_detected = cv2.bitwise_and(frame_detected, mask)

        if self.photo_count < self.number_of_photos:
            self.capture_screenshot(frame_detected)

        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

        cv2.putText(frame, "Target", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        self.generate_alarm("Intruder")

    def capture_screenshot(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/intruder_{timestamp}_{self.photo_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        self.photo_count += 1

    def generate_alarm(self, event_type):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"ALARM: {event_type} detected at {timestamp}")

    def run(self):
        cv2.namedWindow('CCTV Analysis')
        cv2.setMouseCallback('CCTV Analysis', self.draw_polygon)

        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            analyzed_frame = self.analyze_frame(frame)
            cv2.imshow('CCTV Analysis', analyzed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source = "Test Videos/thief_video.mp4"  # Change this to 0 for webcam
    alarm_path = "Alarm/alarm.wav"  # Make sure this path is correct
    analyzer = AdvancedCCTVAnalyzer(video_source, alarm_path)
    analyzer.run()