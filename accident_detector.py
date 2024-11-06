import cv2
import torch
import numpy as np
from datetime import datetime
import os
import pygame

class AccidentDetectionSystem:
    def __init__(self, video_source=0):
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        
        # Initialize YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Create directory for saving incidents
        self.save_dir = "accident_detections"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize pygame for alarm sound
        pygame.mixer.init()
        self.alarm_sound = "Alarm/alarm.wav"
        if os.path.exists(self.alarm_sound):
            try:
                pygame.mixer.music.load(self.alarm_sound)
                print("Alarm sound loaded successfully.")
            except pygame.error as e:
                print(f"Failed to load alarm sound: {e}")
        else:
            print("Alarm sound file not found. Please check the file path.")
        
        # Detection parameters
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        self.previous_positions = {}
        self.object_tracks = {}
        self.track_length = 15  # Increased track length for better pattern analysis
        self.accident_history = {}  # Track potential accidents over time
        
        # Refined thresholds
        self.collision_threshold = 0.4  # Increased for more definitive collisions
        self.speed_threshold = 40  # Increased to detect only significant speed changes
        self.direction_change_threshold = 60  # Increased for more dramatic direction changes
        self.accident_cooldown = 0
        self.cooldown_frames = 30
        
        # New accident detection parameters
        self.min_track_length = 5  # Minimum track length to start analyzing
        self.accident_confidence_threshold = 3  # Number of consecutive frames to confirm accident
        self.deceleration_threshold = 15  # Sudden stopping threshold
        self.minimum_speed_for_accident = 10  # Minimum speed to consider for accident
        
        # Visualization settings
        self.show_tracks = True
        self.show_debug = True

    def calculate_movement(self, current_pos, track):
        """Calculate speed, acceleration, and direction changes"""
        if len(track) < 2:
            return 0, 0, 0
        
        # Calculate instantaneous speed
        speed = np.sqrt(
            (current_pos[0] - track[-1][0])**2 + 
            (current_pos[1] - track[-1][1])**2
        )
        
        # Calculate acceleration (speed change)
        if len(track) >= 3:
            prev_speed = np.sqrt(
                (track[-1][0] - track[-2][0])**2 + 
                (track[-1][1] - track[-2][1])**2
            )
            acceleration = speed - prev_speed
        else:
            acceleration = 0
        
        # Calculate direction change
        if len(track) >= 3:
            prev_direction = np.arctan2(
                track[-1][1] - track[-2][1],
                track[-1][0] - track[-2][0]
            )
            current_direction = np.arctan2(
                current_pos[1] - track[-1][1],
                current_pos[0] - track[-1][0]
            )
            direction_change = abs(np.degrees(current_direction - prev_direction))
            if direction_change > 180:
                direction_change = 360 - direction_change
        else:
            direction_change = 0
            
        return speed, acceleration, direction_change

    def is_actual_accident(self, vehicle_id, movement_data, collision_detected=False):
        """Determine if an actual accident occurred based on multiple factors"""
        speed, acceleration, direction_change = movement_data
        
        # Initialize accident history for new vehicles
        if vehicle_id not in self.accident_history:
            self.accident_history[vehicle_id] = {
                'potential_accident_frames': 0,
                'last_speeds': [],
                'last_accelerations': []
            }
        
        history = self.accident_history[vehicle_id]
        history['last_speeds'].append(speed)
        history['last_accelerations'].append(acceleration)
        
        # Keep only recent history
        history['last_speeds'] = history['last_speeds'][-10:]
        history['last_accelerations'] = history['last_accelerations'][-10:]
        
        # Accident detection criteria
        accident_indicators = 0
        
        # 1. Collision detected
        if collision_detected:
            accident_indicators += 2
        
        # 2. Sudden dramatic deceleration
        if len(history['last_accelerations']) >= 3:
            if np.mean(history['last_accelerations'][-3:]) < -self.deceleration_threshold:
                accident_indicators += 1
        
        # 3. Significant speed and direction change
        if speed > self.minimum_speed_for_accident and (
            speed > self.speed_threshold or 
            direction_change > self.direction_change_threshold
        ):
            accident_indicators += 1
        
        # 4. Sudden stop after high speed
        if len(history['last_speeds']) >= 5:
            avg_speed_before = np.mean(history['last_speeds'][-5:-2])
            current_speed = np.mean(history['last_speeds'][-2:])
            if avg_speed_before > self.minimum_speed_for_accident and current_speed < avg_speed_before * 0.3:
                accident_indicators += 1
        
        # Update potential accident frame counter
        if accident_indicators >= 2:
            history['potential_accident_frames'] += 1
        else:
            history['potential_accident_frames'] = max(0, history['potential_accident_frames'] - 1)
        
        # Determine if it's an actual accident
        return history['potential_accident_frames'] >= self.accident_confidence_threshold

    def detect_accidents(self, frame):
        """Main accident detection method with improved accuracy"""
        results = self.model(frame)
        current_detections = results.pandas().xyxy[0]
        
        accidents_detected = []
        debug_info = []
        current_positions = {}
        
        # Process each detected vehicle
        for _, detection in current_detections.iterrows():
            if detection['name'] in self.vehicle_classes:
                bbox = [
                    int(detection['xmin']), int(detection['ymin']),
                    int(detection['xmax']), int(detection['ymax'])
                ]
                center = (
                    int((bbox[0] + bbox[2]) / 2),
                    int((bbox[1] + bbox[3]) / 2)
                )
                
                vehicle_id = f"{detection['name']}_{len(current_positions)}"
                current_positions[vehicle_id] = {
                    'bbox': bbox,
                    'center': center,
                    'class': detection['name']
                }
                
                # Update tracks
                if vehicle_id not in self.object_tracks:
                    self.object_tracks[vehicle_id] = []
                self.object_tracks[vehicle_id].append(center)
                if len(self.object_tracks[vehicle_id]) > self.track_length:
                    self.object_tracks[vehicle_id].pop(0)
                
                # Calculate movement metrics
                movement_data = self.calculate_movement(
                    center, self.object_tracks[vehicle_id]
                )
                
                # Check for abnormal behavior
                if len(self.object_tracks[vehicle_id]) >= self.min_track_length:
                    collision_detected = False
                    
                    # Check for collisions
                    for other_id, other_pos in current_positions.items():
                        if other_id < vehicle_id:
                            bbox1 = current_positions[vehicle_id]['bbox']
                            bbox2 = other_pos['bbox']
                            
                            # Calculate IoU
                            x_left = max(bbox1[0], bbox2[0])
                            y_top = max(bbox1[1], bbox2[1])
                            x_right = min(bbox1[2], bbox2[2])
                            y_bottom = min(bbox1[3], bbox2[3])
                            
                            if x_right > x_left and y_bottom > y_top:
                                intersection = (x_right - x_left) * (y_bottom - y_top)
                                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                                iou = intersection / (area1 + area2 - intersection)
                                
                                if iou > self.collision_threshold:
                                    collision_detected = True
                                    collision_point = (
                                        int((center[0] + other_pos['center'][0]) / 2),
                                        int((center[1] + other_pos['center'][1]) / 2)
                                    )
                    
                    # Determine if it's an actual accident
                    if self.is_actual_accident(vehicle_id, movement_data, collision_detected):
                        accident_type = 'collision' if collision_detected else 'sudden_change'
                        accident_location = collision_point if collision_detected else center
                        
                        accidents_detected.append({
                            'type': accident_type,
                            'location': accident_location,
                            'vehicle_id': vehicle_id
                        })
                        
                        debug_info.append(f"Accident detected: {accident_type}")
                        debug_info.append(f"Speed={movement_data[0]:.1f}, Direction Change={movement_data[2]:.1f}")
        
        return accidents_detected, current_positions, debug_info

    # The rest of the methods remain the same
    def save_incident(self, frame, incident_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/accident_{incident_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Incident saved: {filename}")

    def draw_visualization(self, frame, accidents, positions, debug_info):
        # Draw vehicle bounding boxes and tracks
        for vehicle_id, data in positions.items():
            color = (0, 255, 0)  # Default color for vehicles
            cv2.rectangle(frame, 
                         (data['bbox'][0], data['bbox'][1]), 
                         (data['bbox'][2], data['bbox'][3]), 
                         color, 2)

            if self.show_tracks and vehicle_id in self.object_tracks:
                track = self.object_tracks[vehicle_id]
                for i in range(len(track) - 1):
                    cv2.line(frame, track[i], track[i + 1], color, 2)
        
        # Draw accident indicators
        for accident in accidents:
            if accident['type'] == 'collision':
                cv2.circle(frame, accident['location'], 50, (0, 0, 255), 2)
                cv2.putText(frame, "COLLISION!", 
                           (accident['location'][0] - 40, accident['location'][1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.circle(frame, accident['location'], 30, (255, 0, 0), 2)
                cv2.putText(frame, "ACCIDENT!", 
                           (accident['location'][0] - 60, accident['location'][1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if self.show_debug:
            for i, info in enumerate(debug_info):
                cv2.putText(frame, info, (10, 30 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def play_alarm(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    def run(self):
        print("Starting Accident Detection System...")
        print("Press 'q' to quit")
        print("Press 'd' to toggle debug visualization")
        print("Press 't' to toggle tracks visualization")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            accidents, positions, debug_info = self.detect_accidents(frame)
            
            if accidents and self.accident_cooldown == 0:
                self.play_alarm()
                for accident in accidents:
                    self.save_incident(frame, accident['type'])
                self.accident_cooldown = self.cooldown_frames
            elif self.accident_cooldown > 0:
                self.accident_cooldown -= 1

            frame = self.draw_visualization(frame, accidents, positions, debug_info)
            cv2.imshow('Accident Detection System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
            elif key == ord('t'):
                self.show_tracks = not self.show_tracks

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    video_path = "Test Videos/acc.mp4"
    detector = AccidentDetectionSystem(video_path)
    detector.run()