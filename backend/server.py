from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import json
import os
from datetime import datetime
import time


app = Flask(__name__)
CORS(app)

os.makedirs('detection_logs', exist_ok=True)
os.makedirs('detection_logs/frames', exist_ok=True)
os.makedirs('detection_logs/metrics', exist_ok=True)


MIN_CONTOUR_AREA = 400
FOCAL_LENGTH_PX = 700
REAL_OBSTACLE_HEIGHT = 1.6
LATENCY_BUDGET_MS = 100
REAL_TIME_MODE = True



class ObstacleTracker:
    """Temporal filtering to smooth detections across frames"""

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.detected_count = 0

    def update(self, centroid_x, centroid_y):
        """Smooth position using Kalman filtering"""
        measurement = np.array([[centroid_x], [centroid_y]], dtype=np.float32)
        self.kalman.correct(measurement)
        predicted = self.kalman.predict()
        self.detected_count += 1
        return predicted[0, 0], predicted[1, 0]

    def get_confidence(self):
        """Higher confidence with consecutive detections"""
        return min(1.0, self.detected_count / 10)

def image_enhancement_fast(img):
    """Fast enhancement for real-time mobile processing"""
    bilateral = cv2.bilateralFilter(img, 5, 50, 50)
    gamma = 0.7
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    enhanced = cv2.LUT(bilateral, lut)
    return enhanced


def detect_obstacles(enhanced_img):
    """Fast detection: Canny edges + contours"""

    if len(enhanced_img.shape) == 3:
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced_img

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    det_bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            det_bboxes.append({
                'bbox': [x, y, x + w, y + h],
                'area': area,
                'solidity': 0.8,
                'aspect_ratio': w / h if h > 0 else 1
            })

    return det_bboxes, edges



def estimate_distance(bbox):
    """Distance using pinhole camera model"""
    x1, y1, x2, y2 = bbox['bbox']
    h = abs(y2 - y1)
    w = abs(x2 - x1)

    if h == 0 or w == 0:
        return None, 0.0

    distance_h = (REAL_OBSTACLE_HEIGHT * FOCAL_LENGTH_PX) / h
    distance_w = (0.4 * FOCAL_LENGTH_PX) / w
    distance = (2 * distance_h + distance_w) / 3
    distance = min(10.0, max(0.1, distance))

    solidity_conf = bbox.get('solidity', 0.5)
    size_conf = min(1.0, bbox.get('area', 0) / 5000)
    overall_confidence = (solidity_conf + size_conf) / 2

    return distance, overall_confidence


def classify_proximity(distance):
    """5-level proximity classification"""
    if distance < 0.5:
        return "CRITICAL"
    elif distance < 1.0:
        return "NEAR"
    elif distance < 2.0:
        return "MID"
    elif distance < 4.0:
        return "FAR"
    else:
        return "CLEAR"


def classify_direction(bbox, frame_width):
    """5-zone direction classification"""
    x1, y1, x2, y2 = bbox['bbox']
    centroid_x = (x1 + x2) / 2
    zone_width = frame_width / 5
    zone = int(centroid_x / zone_width)

    zone_map = {
        0: "FAR_LEFT",
        1: "LEFT",
        2: "CENTER",
        3: "RIGHT",
        4: "FAR_RIGHT"
    }

    return zone_map.get(zone, "CENTER")




def filter_detections_temporal(current_detections, prev_detections):
    """Remove flickering by requiring temporal consistency"""

    if len(prev_detections) == 0:
        return current_detections

    filtered = []
    for det in current_detections:
        is_consistent = False
        for prev in prev_detections:
            dist = np.sqrt((det['bbox'][0] - prev['bbox'][0])**2 +
                          (det['bbox'][1] - prev['bbox'][1])**2)
            if dist < 100:
                is_consistent = True
                break

        if is_consistent or len(prev_detections) == 0:
            filtered.append(det)

    return filtered



def save_frame_for_evaluation(frame, obstacles, frame_id, processing_time_ms):
  

    try:
        frame_small = cv2.resize(frame, (640, 480))
        frame_vis = frame_small.copy()

        for obs in obstacles:
            if 'bbox' in obs:
                x1, y1, x2, y2 = obs['bbox']

                # Color by proximity
                if obs['proximity'] == 'CRITICAL':
                    color = (0, 0, 255)      # Red
                elif obs['proximity'] == 'NEAR':
                    color = (0, 165, 255)    # Orange
                elif obs['proximity'] == 'MID':
                    color = (0, 255, 255)    # Yellow
                else:
                    color = (0, 255, 0)      # Green

                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                label = f"{obs['direction']} {obs['distance']:.1f}m"
                cv2.putText(frame_vis, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save frame with detections
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_file = f"detection_logs/frames/frame_{frame_id:06d}_{timestamp}.jpg"
        cv2.imwrite(frame_file, frame_vis)

        
        metadata = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'processing_time_ms': processing_time_ms,
            'obstacles_detected': len(obstacles),
            'obstacles': [
                {
                    'direction': obs['direction'],
                    'distance': obs['distance'],
                    'proximity': obs['proximity'],
                    'confidence': obs.get('confidence', 0.0),
                    'bbox': obs['bbox']
                } for obs in obstacles
            ]
        }

        # Append to JSONL for batch analysis
        log_file = "detection_logs/realtime_detections.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(metadata) + '\n')

        # Individual metrics per frame
        metrics_file = f"detection_logs/metrics/frame_{frame_id:06d}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"❌ Frame save error: {e}")
        return False



class AdvancedDetector:
    def __init__(self):
        self.tracker = ObstacleTracker()
        self.prev_detections = []
        self.frame_count = 0
        self.metrics = {
            'total_frames': 0,
            'frames_with_obstacles': 0,
            'avg_latency_ms': 0,
            'latency_history': []
        }


detector = AdvancedDetector()


@app.route("/detect", methods=["POST"])
def detect_obstacles_endpoint():
  

    try:
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"📥 Frame #{detector.frame_count + 1}")
        print(f"{'='*70}")

        # Decode image
        data = request.get_json()
        img_b64 = data.get("image")
        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        frame = np.array(img)

        print(f"  📐 Image shape: {frame.shape}")

        # Enhancement
        print(f"  🔧 Enhancing...")
        enhanced = image_enhancement_fast(frame)

        # Detection
        print(f"  🎯 Detecting obstacles...")
        det_bboxes, edges = detect_obstacles(enhanced)

        # Temporal filtering
        det_bboxes = filter_detections_temporal(det_bboxes, detector.prev_detections)

        # Extract features
        print(f"  📏 Estimating distances & directions...")
        obstacles = []

        for det in det_bboxes:
            dist, conf = estimate_distance(det)
            if dist is None:
                continue

            direction = classify_direction(det, frame.shape[1])
            proximity = classify_proximity(dist)

            obstacles.append({
                'direction': direction,
                'distance': round(dist, 2),
                'proximity': proximity,
                'confidence': round(conf, 2),
                'bbox': det['bbox']
            })

        # Sort by distance
        obstacles = sorted(obstacles, key=lambda x: x['distance'])

        # Update state
        detector.prev_detections = det_bboxes
        detector.frame_count += 1

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        detector.metrics['latency_history'].append(latency_ms)
        detector.metrics['total_frames'] += 1
        if obstacles:
            detector.metrics['frames_with_obstacles'] += 1
        detector.metrics['avg_latency_ms'] = np.mean(detector.metrics['latency_history'][-100:])

        print(f"  💾 Logging frame & metrics...")
        save_frame_for_evaluation(frame, obstacles, detector.frame_count, latency_ms)

        if obstacles:
            nearest = obstacles[0]
            audio_message = f"{nearest['proximity']}! Obstacle {nearest['distance']} meters {nearest['direction'].lower()}"
        else:
            audio_message = "Clear path ahead"

       
        response = {
            'detected': len(obstacles) > 0,
            'obstacles': obstacles[:3],
            'frame_id': detector.frame_count,
            'total_detections': len(obstacles),
            'latency_ms': round(latency_ms, 2),
            'avg_latency_ms': round(detector.metrics['avg_latency_ms'], 2),
            'status': 'OK' if latency_ms < LATENCY_BUDGET_MS else 'SLOW',
            'realtime_mode': REAL_TIME_MODE,
            'audio_message': audio_message  
        }

        print(f"  ✅ Response: {len(obstacles)} obstacles, {latency_ms:.1f}ms")
        print(f"  🔊 Audio message: {audio_message}")
        print(f"{'='*70}\n")

        return jsonify(response)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get real-time metrics"""

    return jsonify({
        'total_frames': detector.metrics['total_frames'],
        'frames_with_obstacles': detector.metrics['frames_with_obstacles'],
        'avg_latency_ms': round(detector.metrics['avg_latency_ms'], 2),
        'latency_budget_ms': LATENCY_BUDGET_MS,
        'status': 'FAST' if detector.metrics['avg_latency_ms'] < LATENCY_BUDGET_MS else 'SLOW',
        'detection_rate': round(
            detector.metrics['frames_with_obstacles'] / (detector.metrics['total_frames'] + 1e-6), 2
        )
    })


@app.route("/status", methods=["GET"])
def get_status():
    """Health check"""
    return jsonify({
        'status': 'online',
        'frames_processed': detector.frame_count,
        'realtime_mode': REAL_TIME_MODE,
        'latency_budget_ms': LATENCY_BUDGET_MS
    })


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🌙 CLEAN Detection Server - NO gTTS")
    print(f"⚡ Real-time + Evaluation Pipeline")
    print(f"✅ Backend: Detect + Log")
    print(f"✅ Frontend: Audio (expo-speech)")
    print(f"{'='*70}")
    print(f"\n📱 Real-time Mode: {REAL_TIME_MODE}")
    print(f"⏱  Latency Budget: {LATENCY_BUDGET_MS}ms")
    print(f"🌐 Server: http://0.0.0.0:5000")
    print(f"\n✅ Ready for mobile app\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)