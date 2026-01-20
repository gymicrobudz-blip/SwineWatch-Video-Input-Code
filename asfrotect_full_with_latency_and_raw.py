
# ================= ASFROTECT FULL SYSTEM (WITH LOGGING & RAW SAVE) =================
# NOTE:
# - Adds latency logging to latency_log.csv
# - Saves RAW frames before inference to RAW_SAVE_ROOT
# - Forces thermal sensor restart on math/processing errors

# ------------------- IMPORTS -------------------
import cv2
import numpy as np
import time
import os
import cloudinary
import cloudinary.uploader
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import serial
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, Response, request
import threading
import re
from filterpy.kalman import KalmanFilter
from matplotlib import cm

# ------------------- PATHS -------------------
SAVE_ROOT = "/home/asfrotect/Desktop/detections"
RAW_SAVE_ROOT = "/home/asfrotect/Desktop/raw_frames"
LOG_PATH = "/home/asfrotect/Desktop/latency_log.csv"
PIG_STATE_PATH = "/home/asfrotect/Desktop/pig_state.npy"

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(RAW_SAVE_ROOT, exist_ok=True)

# Initialize latency log
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w") as f:
        f.write("capture_time,upload_time,latency_seconds,image_path,cloud_url\n")

# ------------------- FIREBASE -------------------
cred = credentials.Certificate("/home/asfrotect/Projects/BYMS_TFLITE_3_COMBINED/firebase.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://asfrotect-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

# ------------------- CLOUDINARY -------------------
cloudinary.config(
    cloud_name="dmjhw3xa2",
    api_key="518441677687474",
    api_secret="uWQqwVKVN7QZ_AcU5dWy4htZvR0",
    secure=True
)

# ------------------- THERMAL -------------------
try:
    import board, busio, adafruit_mlx90640
    THERMAL_AVAILABLE = True
except Exception as e:
    print("[THERMAL IMPORT ERROR]", e)
    THERMAL_AVAILABLE = False

THERMAL_MIN_TEMP = 0.0
THERMAL_MAX_TEMP = 0.0
THERMAL_AVG_TEMP = 0.0
thermal_ema = None
THERMAL_ALPHA = 0.2
thermal_lock = threading.Lock()
cached_thermal_resized = None

def reset_thermal():
    global mlx90640, thermal_ema
    try:
        print("[THERMAL] Resetting sensor...")
        time.sleep(1)
        mlx90640 = adafruit_mlx90640.MLX90640(i2c)
        mlx90640.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        thermal_ema = None
        print("[THERMAL] Sensor restarted")
    except Exception as e:
        print("[THERMAL RESET FAILED]", e)

def read_thermal():
    global THERMAL_MIN_TEMP, THERMAL_MAX_TEMP, THERMAL_AVG_TEMP
    try:
        mlx90640.getFrame(thermal_frame)
        data = thermal_frame.reshape((24, 32))

        valid = data[(data > 15.0) & (data < 42.0)]
        if valid.size < 100:
            raise ValueError("Corrupted thermal frame")

        THERMAL_MIN_TEMP = float(np.percentile(valid, 5))
        THERMAL_MAX_TEMP = float(np.percentile(valid, 95))
        THERMAL_AVG_TEMP = float(np.mean(valid))

        return data

    except Exception as e:
        print("[THERMAL ERROR]", e)
        reset_thermal()
        return None

def thermal_to_rgb(t):
    global thermal_ema
    try:
        if thermal_ema is None:
            thermal_ema = t.astype(np.float32)
        else:
            thermal_ema = THERMAL_ALPHA * t + (1 - THERMAL_ALPHA) * thermal_ema

        tmin = max(20.0, THERMAL_MIN_TEMP)
        tmax = min(42.0, THERMAL_MAX_TEMP)
        if tmax - tmin < 0.5:
            raise ValueError("Invalid thermal range")

        norm = np.clip((thermal_ema - tmin) / (tmax - tmin), 0, 1)
        rgb = (cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)
        return rgb

    except Exception as e:
        print("[THERMAL MATH ERROR]", e)
        reset_thermal()
        return None

# ------------------- CAMERA -------------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)},
        controls={"FrameRate": 5}
    )
)
picam2.start()

# ------------------- THERMAL INIT -------------------
if THERMAL_AVAILABLE:
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx90640 = adafruit_mlx90640.MLX90640(i2c)
    mlx90640.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    thermal_frame = np.zeros((24 * 32,), dtype=float)

def thermal_loop():
    global cached_thermal_resized
    while True:
        if THERMAL_AVAILABLE:
            t = read_thermal()
            if t is not None:
                rgb = thermal_to_rgb(t)
                if rgb is not None:
                    rgb = cv2.resize(rgb, (640, 480))
                    with thermal_lock:
                        cached_thermal_resized = rgb
        time.sleep(1)

threading.Thread(target=thermal_loop, daemon=True).start()

# ------------------- CLOUDINARY UPLOAD WITH LOG -------------------
def upload_to_cloudinary(image_path, capture_ts):
    try:
        upload_ts = time.time()
        result = cloudinary.uploader.upload(
            image_path,
            folder=f"asfrotect_detections/{time.strftime('%Y-%m-%d')}",
            resource_type="image"
        )
        latency = upload_ts - capture_ts

        with open(LOG_PATH, "a") as f:
            f.write(f"{capture_ts},{upload_ts},{latency:.3f},{image_path},{result['secure_url']}\n")

        print("[UPLOAD OK]", result["secure_url"], "Latency:", latency)
        return result["secure_url"]

    except Exception as e:
        print("[UPLOAD ERROR]", e)
        return None

# ------------------- MAIN LOOP (SIMPLIFIED CORE) -------------------
while True:
    capture_ts = time.time()
    frame = picam2.capture_array()

    # ---- SAVE RAW FRAME BEFORE INFERENCE ----
    date = time.strftime("%Y-%m-%d")
    raw_dir = os.path.join(RAW_SAVE_ROOT, date)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"{int(capture_ts)}.jpg")
    cv2.imwrite(raw_path, frame)

    # ---- SIMULATED DETECTION RESULT (PLACEHOLDER) ----
    detected = frame.copy()

    # ---- COMBINE WITH THERMAL ----
    with thermal_lock:
        thermal_img = cached_thermal_resized
    if thermal_img is not None:
        detected = cv2.hconcat([detected, thermal_img])

    save_dir = os.path.join(SAVE_ROOT, date)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{int(capture_ts)}_det.jpg")
    cv2.imwrite(save_path, detected)

    upload_to_cloudinary(save_path, capture_ts)

    time.sleep(60)
