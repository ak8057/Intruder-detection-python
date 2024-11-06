import cv2
import torch
import numpy as np
import pygame
from datetime import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import time


class AdvancedCCTVAnalyzer:
    def __init__(self, video_source, alarm_path="alarm.wav", cascade_path="fire_detection_cascade_model.xml", 

                 email_config=None):
        self.video = cv2.VideoCapture(video_source)
        self.alarm_path = alarm_path
        self.screenshot_dir = "intruder_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)

    # Initialize pygame for audio
        pygame.mixer.init()
        pygame.mixer.music.load(self.alarm_path)

    # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load the fire detection cascade
        self.fire_cascade = cv2.CascadeClassifier(cascade_path)

        self.target_classes = ['person', 'car', 'truck', 'bus']
        self.pts = []  # Polygon points for ROI
        self.number_of_photos = 3
        self.photo_count = 0
    # Fire detection parameters
        self.fire_threshold = 0.3  # Minimum ratio of fire-colored pixels
        self.fire_detection_cooldown = 10  # Frames between fire checks
        self.current_cooldown = 0
        self.consecutive_detections = 0
        self.required_consecutive = 3  # Number of consecutive detections needed

    # Email configuration

        self.email_config = email_config or {

            'smtp_server': 'smtp.gmail.com',

            'smtp_port': 587,

            'sender_email': 'abhay29032005@gmail.com',

            'sender_password': 'vren twft bfyg tzgi',

            'recipient_email': 'guptavishnu2711@gmail.com'

        }

        

        # Alert tracking

        self.last_fire_alert = 0

        self.last_intruder_alert = 0

        self.alert_cooldown = 3600  # 1 hour cooldown between alerts



    def send_email_alert(self, event_type, image_path=None):

        """

        Send email alert with optional attachment

        """

        current_time = time.time()

        

        # Check cooldown

        if event_type == "Fire" and current_time - self.last_fire_alert < self.alert_cooldown:

            return

        if event_type == "Intruder" and current_time - self.last_intruder_alert < self.alert_cooldown:

            return



        try:

            msg = MIMEMultipart()

            msg['From'] = self.email_config['sender_email']

            msg['To'] = self.email_config['recipient_email']

            msg['Subject'] = f'ALERT: {event_type} Detected!'



            # Email body

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            body = f"""

            Security Alert!

            

            Event Type: {event_type}

            Timestamp: {timestamp}

            Location: CCTV Camera

            

            Please check your security system immediately.

            """

            msg.attach(MIMEText(body, 'plain'))



            # Attach image if available

            if image_path and os.path.exists(image_path):

                with open(image_path, 'rb') as f:

                    img_data = f.read()

                    image = MIMEImage(img_data)

                    image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))

                    msg.attach(image)



            # Connect to SMTP server and send email

            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:

                server.starttls()

                server.login(self.email_config['sender_email'], self.email_config['sender_password'])

                server.send_message(msg)



            # Update last alert time

            if event_type == "Fire":

                self.last_fire_alert = current_time

            else:

                self.last_intruder_alert = current_time



            print(f"Email alert sent for {event_type} detection")



        except Exception as e:

            print(f"Failed to send email alert: {str(e)}")



    def draw_polygon(self, event, x, y, flags, param):
        """Handle mouse events for drawing the ROI polygon"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pts = []

    def inside_polygon(self, point, polygon):
        """Check if a point is inside the defined polygon"""
        if len(polygon) > 0:
            return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0
        return False

    def preprocess(self, img):
        """Preprocess the input frame"""
        height, width = img.shape[:2]
        ratio = height / width
        return cv2.resize(img, (640, int(640 * ratio)))
        
    def detect_fire(self, frame):
        """
        Fire detection using Haar/LBP Cascade and YOLOv5 (if available).
        Returns: (bool, frame) - fire detected flag and annotated frame
        """
        fire_detected = False

    # Detect fire-like regions with cascade
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fire_regions = self.fire_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        for (x, y, w, h) in fire_regions:
            fire_detected = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Fire Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Additional confirmation with YOLO (optional)
        if fire_detected and self.model:
            results = self.model(frame)
            detections = results.pandas().xyxy[0]
            for _, row in detections.iterrows():
                if row['name'] in ['smoke', 'fire', 'flame']:
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Fire Confirmed!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    fire_detected = True

        return fire_detected, frame


    def analyze_frame(self, frame):
        frame_detected = frame.copy()
        frame = self.preprocess(frame)
        
        # Fire detection with cooldown
        if self.current_cooldown <= 0:
            fire_detected, frame = self.detect_fire(frame)
            if fire_detected:
                self.handle_fire(frame_detected)
                self.current_cooldown = self.fire_detection_cooldown
        else:
            self.current_cooldown -= 1

        # Object detection
        results = self.model(frame)
        results = results.pandas().xyxy[0]

        for index, row in results.iterrows():
            if row['name'] in self.target_classes:
                name = str(row['name'])
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                if len(self.pts) >= 4 and self.inside_polygon((center_x, center_y), np.array([self.pts])) and name == 'person':
                    self.handle_intruder(frame, frame_detected, x1, y1, x2, y2, center_x, center_y)

        # Draw polygon if points are set
        if len(self.pts) >= 4:
            frame_copy = frame.copy()
            cv2.fillPoly(frame_copy, np.array([self.pts]), (0, 255, 0))
            frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)

        return frame

    def handle_intruder(self, frame, frame_detected, x1, y1, x2, y2, center_x, center_y):
        """Handle intruder detection"""
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

    def handle_fire(self, frame):
        """Handle fire detection"""
        screenshot_path = self.capture_screenshot(frame)

            # Send email with the first screenshot

        if self.photo_count == 1:

                self.send_email_alert("Fire", screenshot_path)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

        self.generate_alarm("Fire")

    def capture_screenshot(self, frame):
        """Capture and save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/intruder_{timestamp}_{self.photo_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        self.photo_count += 1
        if self.photo_count >= self.number_of_photos:
            self.photo_count = 0

    def generate_alarm(self, event_type):
        """Generate alarm with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("Alarm.txt", "a") as f:

            f.write(f"{timestamp} - {event_type} Detected\n")



    def run(self):
        """Main run loop"""
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
    # video_source = "Test Videos/fire.mp4"
    # video_source = "Test Videos/fire4.mp4"
    # video_source = "Test Videos/fire5.mp4"
    video_source = 0
    # video_source = "Test Videos/thief_video.mp4"
    alarm_path = "Alarm/alarm.wav"
    analyzer = AdvancedCCTVAnalyzer(video_source, alarm_path)
    analyzer.run()