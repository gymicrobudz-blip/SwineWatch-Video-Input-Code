#!/usr/bin/env python3
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response
from picamera2 import Picamera2
from libcamera import controls

# ================== OPTIONAL THERMAL ==================
try:
    import board, busio, adafruit_mlx90640
    from matplotlib import cm
    THERMAL_AVAILABLE = True
except Exception as e:
    print("Thermal not available:", e)
    THERMAL_AVAILABLE = False

# ================== MODELS ==================
PIG_MODEL_PATH      = "/home/asfrotect/Projects/pig_vs_non_pig.v3.pt"
ASF_MODEL_PATH      = "/home/asfrotect/Projects/best(ASF_MODEL)_.pt"
BEHAVIOR_MODEL_PATH = "/home/asfrotect/Projects/best(BEHAVIOR_MODEL)_.pt"

pig_model      = YOLO(PIG_MODEL_PATH)
asf_model      = YOLO(ASF_MODEL_PATH)
behavior_model = YOLO(BEHAVIOR_MODEL_PATH)

for m in (pig_model, asf_model, behavior_model):
    m.overrides["agnostic_nms"] = True

# ================== CAMERA ==================
W, H = 640, 480
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (W, H)},
    controls={"AwbEnable": True, "AwbMode": controls.AwbModeEnum.Auto}
)
picam2.configure(cfg)
picam2.start()
time.sleep(1.5)

# ================== THERMAL ==================
if THERMAL_AVAILABLE:
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    thermal_buf = np.zeros((24 * 32,), dtype=np.float32)

def read_thermal():
    mlx.getFrame(thermal_buf)
    return thermal_buf.reshape(24, 32)

def thermal_to_rgb(data, tmin=20, tmax=40):
    norm = np.clip((data - tmin) / (tmax - tmin), 0, 1)
    return (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)

# ================== HELPERS ==================
def draw(frame, box, text, color):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    cv2.putText(frame,text,(x1,y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

# ================== TIMERS ==================
ASF_INTERVAL = 6
BEH_INTERVAL = 6
asf_ctr = beh_ctr = 0

last_lesions  = []
last_behavior = None

# ================== FLASK ==================
app = Flask(__name__)
client = False

def generate_rgb():
    global asf_ctr, beh_ctr, last_lesions, last_behavior, client
    prev = time.time()

    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 0) 
        # -------- PIG DETECTION --------
        pigs = []
        res = pig_model(frame, conf=0.4, stream=True)
        for r in res:
            for b in r.boxes:
                if "pig" in r.names[int(b.cls[0])].lower():
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    pigs.append((x1,y1,x2,y2))

        # -------- ASF --------
        if asf_ctr == 0:
            last_lesions = []
            for (x1,y1,x2,y2) in pigs:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: continue
                ar = asf_model(roi, conf=0.5, stream=True)
                for r in ar:
                    for b in r.boxes:
                        if b.conf[0] >= 0.5:
                            lx1,ly1,lx2,ly2 = map(int,b.xyxy[0])
                            last_lesions.append(
                                (x1+lx1,y1+ly1,x1+lx2,y1+ly2)
                            )
        asf_ctr = (asf_ctr+1) % ASF_INTERVAL

        # -------- BEHAVIOR --------
        if beh_ctr == 0:
            last_behavior = None
            for (x1,y1,x2,y2) in pigs:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: continue
                br = behavior_model(roi, conf=0.5, stream=True)
                for r in br:
                    for b in r.boxes:
                        if b.conf[0] >= 0.5:
                            last_behavior = r.names[int(b.cls[0])]
                            break
        beh_ctr = (beh_ctr+1) % BEH_INTERVAL

        # -------- DRAW --------
        for box in pigs:
            lbl = "pig"
            if last_behavior:
                lbl += f" | {last_behavior}"
            draw(frame, box, lbl, (0,255,0))

        for l in last_lesions:
            draw(frame, l, "ASF", (0,0,255))

        # -------- FPS --------
        now = time.time()
        fps = 1/(now-prev)
        prev = now
        cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        if client:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")

def generate_thermal():
    while True:
        if not THERMAL_AVAILABLE:
            img = np.zeros((H,W,3),np.uint8)
            cv2.putText(img,"Thermal N/A",(50,240),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        else:
            t = read_thermal()
            img = cv2.resize(thermal_to_rgb(t),(W,H))
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

# ================== ROUTES ==================
@app.route("/")
def index():
    return """
    <h2>RGB</h2><img src='/rgb'>
    <h2>Thermal</h2><img src='/thermal'>
    """

@app.route("/rgb")
def rgb():
    global client
    client = True
    return Response(generate_rgb(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/thermal")
def thermal():
    global client
    client = True
    return Response(generate_thermal(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
