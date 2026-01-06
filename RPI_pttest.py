#!/usr/bin/env python3
import os, time, threading, json, sys
import cv2, numpy as np
from flask import Flask, Response, request, abort
from picamera2 import Picamera2
import board, busio, adafruit_mlx90640
from ultralytics import YOLO

print("ASFROTECT STREAMER â€“ PiCam + Thermal + Inference + UI")

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

MODEL_ASF_PATH = "/home/asfrotect/Projects/best(ASF_MODEL)_.pt"
MODEL_BEHAVIOR_PATH = "/home/asfrotect/Projects/best(BEHAVIOR_MODEL)_.pt"
MODEL_PIG_PATH = "/home/asfrotect/Projects/pig_vs_non_pig.v3.pt"

for path in [MODEL_ASF_PATH, MODEL_BEHAVIOR_PATH, MODEL_PIG_PATH]:
    if not os.path.exists(path):
        print(f"âŒ Missing model: {path}")
        sys.exit(1)

# Load YOLO models
model_asf = YOLO(MODEL_ASF_PATH)
model_behavior = YOLO(MODEL_BEHAVIOR_PATH)
model_pig = YOLO(MODEL_PIG_PATH)

# Force CPU + float32 only
for m in (model_asf, model_behavior, model_pig):
    try:
        m.to("cpu")
    except:
        pass
    try:
        m.model.float()  # ensure FP32
    except:
        pass

# Camera tuning
CAM_SIZE = (640, 360)
CAM_FORMAT = "RGB888"
JPEG_QUALITY = 70
IMG_SZ_PIG = 320
IMG_SZ_CROP = 320

# Thresholds
PIG_CONF_THRESH = 0.5
ASFF_CONF_THRESH = 0.5
BEHAV_CONF_THRESH = 0.5

# Thermal system parameters
THERM_TMIN, THERM_TMAX = 20.0, 45.0
SCALE_DEFAULT = 16
COLORMAP = cv2.COLORMAP_INFERNO

STREAM_TOKEN = os.environ.get("STREAM_TOKEN", "")

thermal_lock = threading.Lock()  # Ensures safe access to thermal data
app = Flask(__name__)

def _enforce_token():
    if STREAM_TOKEN and request.args.get("token") != STREAM_TOKEN:
        abort(403)

# -----------------------------------------------------------
# THERMAL CAMERA INITIALIZATION (MLX90640)
# -----------------------------------------------------------

# Initialize the thermal sensor (assuming MLX90640)
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)

# Set up the sensor parameters
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ  # Set refresh rate
thermal_latest = {"frame": None}

def read_thermal_data():
    while True:
        try:
            # Read thermal data into a 768-length array (24x32 sensor)
            thermal_data = np.zeros(768)
            mlx.getFrame(thermal_data)
            
            # Reshape it into a 2D array (24x32)
            thermal_data = thermal_data.reshape((24, 32))

            # Update the latest thermal frame data
            with thermal_lock:
                thermal_latest["frame"] = thermal_data

        except Exception as e:
            print("Error reading thermal data:", e)

        time.sleep(1 / 16)  # Delay based on the refresh rate (16 Hz)

# Start the thermal data reading thread
threading.Thread(target=read_thermal_data, daemon=True).start()

# -----------------------------------------------------------
# CAMERA INITIALIZATION (Picamera2)
# -----------------------------------------------------------

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": CAM_SIZE, "format": CAM_FORMAT},
    controls={"FrameDurationLimits": (33333, 33333),
              "AwbEnable": True,
              "Brightness": 0.0,
              "Contrast": 1.0,
              "Saturation": 1.0}
)
picam2.configure(config)
picam2.start()

_cam_lock = threading.Lock()
latest_frame = {"data": None}
annotated_shared = {"img": None, "detections": []}
annotated_lock = threading.Lock()

# Example of ensuring proper frame capture and locks:
def _picam_reader():
    while True:
        try:
            with _cam_lock:
                frame = picam2.capture_array()  # Corrected frame capture call
                if frame is None:
                    print("Received empty frame, retrying")
                    time.sleep(0.05)
                    continue
            latest_frame["data"] = frame
        except Exception as e:
            print("Error reading from camera:", e)
            time.sleep(0.05)

threading.Thread(target=_picam_reader, daemon=True).start()

# -----------------------------------------------------------
# DRAWING UTILITIES
# -----------------------------------------------------------

def draw_box_label(img, box, label, color=(0,255,0), scale=0.5, thickness=1):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
    cv2.putText(img, label, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness, cv2.LINE_AA)

def class_name_contains_pig(name):
    try:
        return "pig" in name.lower()
    except:
        return False

# -----------------------------------------------------------
# MJPEG CAMERA STREAM
# -----------------------------------------------------------

# FPS Calculation for Camera Stream
def gen_cam():
    while True:
        with _cam_lock:
            frame = latest_frame.get("data")
        
        if frame is None:
            print("No frame available, retrying...")
            time.sleep(0.05)
            continue

        # Resize frame (optional)
        resized_frame = cv2.resize(frame, (640, 480))

        # Pig detection
        pig_results = model_pig(resized_frame, stream=False, conf=PIG_CONF_THRESH)
        pigs = []
        for r in pig_results:
            for b in r.boxes:
                conf = float(b.conf[0])
                if conf < PIG_CONF_THRESH:
                    continue
                cls_id = int(b.cls[0])
                name = r.names[cls_id]
                if "pig" not in name.lower():
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                pigs.append((x1, y1, x2, y2, conf))

        # ASF and Behavior detection for each pig
        for x1, y1, x2, y2, conf in pigs:
            pig_roi = resized_frame[y1:y2, x1:x2]
            
            # ASF inference
            asf_results = model_asf(pig_roi, conf=ASFF_CONF_THRESH, stream=False)
            for ar in asf_results:
                for abox in ar.boxes:
                    conf2 = float(abox.conf[0])
                    if conf2 < ASFF_CONF_THRESH:
                        continue
                    cls_id = int(abox.cls[0])
                    name = ar.names[cls_id].lower()
                    ax1, ay1, ax2, ay2 = map(int, abox.xyxy[0])
                    ax1 += x1; ay1 += y1
                    ax2 += x1; ay2 += y1
                    draw_box_label(resized_frame, (ax1, ay1, ax2, ay2), f"ASF: {name}", color=(0, 0, 255))

            # Behavior inference
            beh_results = model_behavior(pig_roi, conf=BEHAV_CONF_THRESH, stream=False)
            for br in beh_results:
                for bbox in br.boxes:
                    if float(bbox.conf[0]) >= BEHAV_CONF_THRESH:
                        behavior_label = br.names[int(bbox.cls[0])]
                        draw_box_label(resized_frame, (x1, y1, x2, y2), f"{behavior_label}", color=(255, 255, 0))

        # Encode frame
        ret, buf = cv2.imencode('.jpg', resized_frame)
        if ret:
            frame = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# -----------------------------------------------------------
# MJPEG STREAM FOR CAMERA FEED
# -----------------------------------------------------------

@app.route('/camera_feed')
def camera_feed():
    _enforce_token()
    return Response(gen_cam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------
# THERMAL STREAM (MJPEG)
# -----------------------------------------------------------

def gen_thermal():
    while True:
        with thermal_lock:
            thermal_data = thermal_latest.get("frame")
        
        if thermal_data is None:
            time.sleep(0.05)
            continue
        
        # Normalize thermal data to the given range
        thermal_img = np.interp(thermal_data, (THERM_TMIN, THERM_TMAX), (0, 255))
        thermal_img = thermal_img.astype(np.uint8)

        # Apply color map for visualization
        thermal_img = cv2.applyColorMap(thermal_img, COLORMAP)

        # Encode thermal image as MJPEG frame
        ret, buf = cv2.imencode('.jpg', thermal_img)
        if ret:
            frame = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/thermal_feed')
def thermal_feed():
    _enforce_token()
    return Response(gen_thermal(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------
# COMBINED STREAM VIEW (Camera + Thermal)
# -----------------------------------------------------------

# -----------------------------------------------------------
# COMBINED STREAM VIEW (Camera + Thermal)
# -----------------------------------------------------------

@app.route('/view_combined')
def view_combined():
    # Check the token inside the request context
    _enforce_token()
    
    while True:
        with _cam_lock, thermal_lock:
            frame = latest_frame.get("data")
            thermal_frame = thermal_latest.get("frame")
        
        if frame is None or thermal_frame is None:
            time.sleep(0.05)
            continue
        
        # Normalize and apply color map to thermal data
        thermal_img = np.interp(thermal_frame, (THERM_TMIN, THERM_TMAX), (0, 255))
        thermal_img = thermal_img.astype(np.uint8)
        thermal_img = cv2.applyColorMap(thermal_img, COLORMAP)

        # Resize both frames to same size
        frame_resized = cv2.resize(frame, (640, 360))
        thermal_resized = cv2.resize(thermal_img, (640, 360))

        # Combine both frames side by side
        combined = np.hstack((frame_resized, thermal_resized))

        # Encode combined image
        ret, buf = cv2.imencode('.jpg', combined)
        if ret:
            frame = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# -----------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
