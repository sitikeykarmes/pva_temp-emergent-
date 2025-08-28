import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import time
import os
import pickle
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier

class ParkingDetectionSystem:
    def __init__(self):
        # Load the YOLO model
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Define object categories and weights
        self.object_weights = {
            'car': 5,          # Parking indicator
            'motorcycle': 6,    # Parking indicator
            'parking meter': 6, # Parking indicator
            'bus': 5,          # Parking indicator (large vehicle)
            'truck': 5,         # Parking indicator (large vehicle)
            'bicycle': 3,       # Parking indicator (can be parked)
            'road': -2,       # No parking indicator
            'person': -1,    # No parking indicator (pedestrian areas)
            'building': -0.5,   # No parking indicator
            'footpath': -0.5,   # No parking indicator
            'sign': -0.25,     # No parking indicator (could be a no-parking sign)
            'traffic light': -0.5, # No parking indicator (near intersections)
            'fire hydrant': -1, # No parking indicator (strictly prohibited)
            'stop sign': -1,    # No parking indicator (near intersections)
            'parking area': 5,  # Parking indicator (designated parking zones)
            'no-parking sign': -3  # Strong no parking indicator
        }
        
        # Initialize device and CNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = self._init_cnn_model()
        
        # Image transform for CNN input
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Vehicle tracker
        self.tracker = VehicleTracker(iou_threshold=0.3)
        
        # Persistent alerts
        self.persistent_alerts = {}
        
        # Load or create classifier model
        self.classifier_model = self._load_or_create_model()
        
    def _init_cnn_model(self):
        """Initialize CNN Feature Extractor"""
        class CNNFeatureExtractor(nn.Module):
            def __init__(self):
                super(CNNFeatureExtractor, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

                self.fc_layers = nn.Sequential(
                    nn.Linear(64 * 16 * 16, 128),
                    nn.ReLU(),
                    nn.Linear(128, 6)  # 6 additional deep features
                )

            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc_layers(x)
                return x
        
        model = CNNFeatureExtractor().to(self.device)
        model.eval()
        return model
    
    def _load_or_create_model(self):
        """Load existing model or create a new one with default parameters"""
        try:
            # For demo purposes, create a basic model
            # In production, you would load from your trained model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Create dummy training data for initialization
            dummy_features = np.random.rand(10, 12)  # 12 features
            dummy_labels = np.random.choice([0, 1], 10)  # 0: Parking, 1: No Parking
            
            model.fit(dummy_features, dummy_labels)
            return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            return None
    
    def extract_features_from_frame(self, frame):
        """Extract features from a video frame"""
        try:
            # YOLO detection
            results = self.yolo_model(frame)
            detected_objects = [self.yolo_model.names[int(box.cls)] for box in results[0].boxes]

            parking_score = 0
            for obj in detected_objects:
                parking_score += self.object_weights.get(obj, 0)

            # Feature Engineering
            img_area = frame.shape[0] * frame.shape[1]

            vehicle_area = 0
            for box in results[0].boxes:
                if self.yolo_model.names[int(box.cls)] in ['car', 'motorcycle', 'bus', 'truck', 'bicycle']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_area += (x2 - x1) * (y2 - y1)

            area_ratio = vehicle_area / img_area if img_area > 0 else 0
            object_density = len(detected_objects) / img_area if img_area > 0 else 0

            # Color Features
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mean_hue = np.mean(hsv[:, :, 0])
            mean_saturation = np.mean(hsv[:, :, 1])
            mean_value = np.mean(hsv[:, :, 2])

            # CNN Deep Feature Extraction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb_frame)
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                cnn_features = self.cnn_model(image_tensor).cpu().numpy().flatten()

            return [parking_score, area_ratio, object_density, mean_hue, mean_saturation, mean_value] + list(cnn_features)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return [0] * 12  # Return default features
    
    def predict_from_frame(self, frame):
        """Predict parking zone from a frame"""
        try:
            if self.classifier_model is None:
                # Fallback prediction based on parking score
                features = self.extract_features_from_frame(frame)
                parking_score = features[0]
                return "Parking Zone" if parking_score > 0 else "No Parking Zone"
            
            features = self.extract_features_from_frame(frame)
            
            # Create DataFrame with features
            column_names = ['parking_score', 'area_ratio', 'object_density', 'mean_hue', 
                           'mean_saturation', 'mean_value'] + [f'cnn_feature_{i}' for i in range(6)]
            
            input_data = pd.DataFrame([features], columns=column_names)
            
            # Make prediction
            prediction = self.classifier_model.predict(input_data)[0]
            return "Parking Zone" if prediction == 0 else "No Parking Zone"
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Unknown Zone"
    
    def get_vehicle_detections(self, frame):
        """Get vehicle detections from frame"""
        results = self.yolo_model(frame)
        vehicles = []
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.yolo_model.names[cls]
            
            if class_name in ['car', 'motorcycle', 'bus', 'truck', 'bicycle'] and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicles.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': conf
                })
        
        return vehicles
    
    def process_frame(self, frame, alert_threshold_seconds=5):
        """Process a single frame and return detection results"""
        current_time = time.time()
        
        # Predict zone
        prediction = self.predict_from_frame(frame)
        
        # Get vehicle detections
        vehicles = self.get_vehicle_detections(frame)
        vehicle_boxes = [v['bbox'] for v in vehicles]
        
        # Update vehicle tracking
        is_no_parking_zone = (prediction == "No Parking Zone")
        tracked_vehicles = self.tracker.update(vehicle_boxes, current_time, is_no_parking_zone)
        
        # Process alerts
        alerts = []
        for vehicle_box, vehicle_id, first_detection_time, continuous_detection_time in tracked_vehicles:
            if is_no_parking_zone and continuous_detection_time > alert_threshold_seconds:
                if vehicle_id not in self.persistent_alerts:
                    alert_text = f"VIOLATION: Vehicle #{vehicle_id} in no-parking zone for {continuous_detection_time:.1f}s"
                    self.persistent_alerts[vehicle_id] = {
                        'text': alert_text,
                        'bbox': vehicle_box,
                        'timestamp': current_time,
                        'duration': continuous_detection_time
                    }
                    alerts.append(self.persistent_alerts[vehicle_id])
                else:
                    # Update existing alert
                    self.persistent_alerts[vehicle_id]['duration'] = continuous_detection_time
        
        # Prepare response
        response = {
            'prediction': prediction,
            'vehicles': [],
            'alerts': list(self.persistent_alerts.values()),
            'timestamp': current_time
        }
        
        # Add vehicle info with tracking data
        for i, (vehicle_box, vehicle_id, first_detection_time, continuous_detection_time) in enumerate(tracked_vehicles):
            vehicle_data = vehicles[i] if i < len(vehicles) else {'class': 'vehicle', 'confidence': 0.8}
            
            response['vehicles'].append({
                'id': vehicle_id,
                'bbox': vehicle_box,
                'class': vehicle_data['class'],
                'confidence': vehicle_data['confidence'],
                'duration': continuous_detection_time,
                'status': 'violation' if (is_no_parking_zone and continuous_detection_time > alert_threshold_seconds) else 'normal'
            })
        
        return response
    
    def reset_alerts(self):
        """Reset all persistent alerts"""
        self.persistent_alerts.clear()


def calculate_iou(box1, box2):
    """Calculate intersection over union between two bounding boxes"""
    x1_intr = max(box1[0], box2[0])
    y1_intr = max(box1[1], box2[1])
    x2_intr = min(box1[2], box2[2])
    y2_intr = min(box1[3], box2[3])
    
    w_intr = max(0, x2_intr - x1_intr)
    h_intr = max(0, y2_intr - y1_intr)
    area_intr = w_intr * h_intr
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = area_intr / float(area_box1 + area_box2 - area_intr)
    return iou


class VehicleTracker:
    def __init__(self, iou_threshold=0.5):
        self.tracked_vehicles = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
    
    def update(self, current_boxes, current_time, is_no_parking_zone):
        if not current_boxes:
            for vehicle in self.tracked_vehicles:
                if is_no_parking_zone:
                    vehicle[3] = 0
            return []
        
        updated_vehicles = []
        matched_indices = set()
        
        for i, vehicle in enumerate(self.tracked_vehicles):
            tracked_box, vehicle_id, first_detection_time, continuous_detection_time = vehicle
            
            best_match_idx = -1
            best_iou = self.iou_threshold
            
            for j, current_box in enumerate(current_boxes):
                if j in matched_indices:
                    continue
                
                iou = calculate_iou(tracked_box, current_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j
            
            if best_match_idx >= 0:
                matched_indices.add(best_match_idx)
                updated_box = current_boxes[best_match_idx]
                
                if is_no_parking_zone:
                    continuous_detection_time += current_time - first_detection_time
                    first_detection_time = current_time
                else:
                    continuous_detection_time = 0
                    first_detection_time = current_time
                
                updated_vehicles.append([updated_box, vehicle_id, first_detection_time, continuous_detection_time])
        
        for j, current_box in enumerate(current_boxes):
            if j not in matched_indices:
                vehicle_id = self.next_id
                self.next_id += 1
                continuous_detection_time = 0
                updated_vehicles.append([current_box, vehicle_id, current_time, continuous_detection_time])
        
        self.tracked_vehicles = updated_vehicles
        return self.tracked_vehicles