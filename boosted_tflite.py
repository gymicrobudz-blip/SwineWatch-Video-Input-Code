import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# ================= PERFORMANCE TWEAKS =================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ================= CONFIG =================
PIG_INPUT_SIZE = 416
BEHAVIOR_INPUT_SIZE = 256
SKIN_INPUT_SIZE = 320
SMALL_W = 320

PIG_MODEL = "/home/asfrotect/Projects/BYMS - TFLITE 3 COMBINED/PigvsNONPig-v2_float16.tflite"
BEHAVIOR_MODEL = "/home/asfrotect/Projects/BYMS - TFLITE 3 COMBINED/bestv8_behavior-2_float16.tflite"
SKIN_MODEL = "/home/asfrotect/Projects/BYMS - TFLITE 3 COMBINED/ASFskin_NoPartsv3_v8s.tflite"

PIG_CLASS_ID = 7
HUMAN_CLASS_ID = 0

PIG_CONF = 0.50
SKIN_CONF = 0.45

BEHAVIOR_INTERVAL = 6
SKIN_INTERVAL = 5

# ================= LABELS =================
BEHAVIOR_NAMES = {0: "ACTIVE", 1: "EATING", 2: "GROUP", 3: "INACTIVE"}
SKIN_NAMES = {1: "ASF LESION", 2: "REDNESS"}

# ================= PREPROCESS =================
def preprocess_pig(img):
    img = cv2.resize(img, (PIG_INPUT_SIZE, PIG_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)[None]

def preprocess_behavior(img):
    img = cv2.resize(img, (BEHAVIOR_INPUT_SIZE, BEHAVIOR_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)[None]

def preprocess_skin(img):
    img = cv2.resize(img, (SKIN_INPUT_SIZE, SKIN_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)[None]

# ================= HELPERS =================
def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def overlaps_any(box, boxes, thr=0.25):
    return any(compute_iou(box, b) > thr for b in boxes)

def nms(dets, thr):
    dets = sorted(dets, key=lambda x: x[1], reverse=True)
    out = []
    while dets:
        best = dets.pop(0)
        out.append(best)
        dets = [d for d in dets if compute_iou(best[0], d[0]) < thr]
    return out

def load_model(path):
    itp = Interpreter(model_path=path, num_threads=4)
    itp.allocate_tensors()
    return itp

# ================= MODELS =================
pig_itp = load_model(PIG_MODEL)
beh_itp = load_model(BEHAVIOR_MODEL)
skin_itp = load_model(SKIN_MODEL)

pig_in, pig_out = pig_itp.get_input_details(), pig_itp.get_output_details()
beh_in, beh_out = beh_itp.get_input_details(), beh_itp.get_output_details()
skin_in, skin_out = skin_itp.get_input_details(), skin_itp.get_output_details()

# ================= CAMERA =================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()

frame_id = 0
fps_t = time.time()
last_behavior = "INACTIVE"

# ================= MAIN LOOP =================
while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 0)
    frame_id += 1
    H, W, _ = frame.shape

    # ---- PIG DETECTOR ----
    small = cv2.resize(frame, (SMALL_W, int(H * (SMALL_W / W))))
    sx, sy = W / small.shape[1], H / small.shape[0]

    pigs = []
    humans = []

    pig_itp.set_tensor(pig_in[0]['index'], preprocess_pig(small))
    pig_itp.invoke()
    preds = pig_itp.get_tensor(pig_out[0]['index'])[0].T

    for d in preds:
        x, y, bw, bh = d[:4]
        scores = d[4:]
        cid = int(np.argmax(scores))
        conf = float(scores[cid])
        if conf < PIG_CONF:
            continue

        x1 = int((x - bw / 2) * small.shape[1] * sx)
        y1 = int((y - bh / 2) * small.shape[0] * sy)
        x2 = int((x + bw / 2) * small.shape[1] * sx)
        y2 = int((y + bh / 2) * small.shape[0] * sy)

        if cid == PIG_CLASS_ID:
            pigs.append(((x1, y1, x2, y2), conf))
        elif cid == HUMAN_CLASS_ID:
            humans.append((x1, y1, x2, y2))

    pigs = nms(pigs, 0.4)

    # ---- BEHAVIOR (BEST PIG ONLY) ----
    if pigs and frame_id % BEHAVIOR_INTERVAL == 0:
        (x1, y1, x2, y2), _ = pigs[0]
        roi = frame[y1:y2, x1:x2]
        if roi.size:
            beh_itp.set_tensor(
                beh_in[0]['index'],
                preprocess_behavior(roi)
            )
            beh_itp.invoke()
            o = beh_itp.get_tensor(beh_out[0]['index'])[0].T
            last_behavior = BEHAVIOR_NAMES[
                int(np.argmax(max(o, key=lambda d: max(d[4:]))[4:]))
            ]

    for (x1, y1, x2, y2), conf in pigs:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"PIG {conf:.2f} | {last_behavior}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    # ---- SKIN (PIG ROI ONLY) ----
    if pigs and frame_id % SKIN_INTERVAL == 0:
        for (x1, y1, x2, y2), _ in pigs:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            skin_itp.set_tensor(
                skin_in[0]['index'],
                preprocess_skin(roi)
            )
            skin_itp.invoke()
            outs = skin_itp.get_tensor(skin_out[0]['index'])[0].T

            for d in outs:
                x, y, bw, bh = d[:4]
                scores = d[4:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if cid not in SKIN_NAMES or conf < SKIN_CONF:
                    continue

                rx1 = int(x1 + (x - bw / 2) * (x2 - x1))
                ry1 = int(y1 + (y - bh / 2) * (y2 - y1))
                rx2 = int(x1 + (x + bw / 2) * (x2 - x1))
                ry2 = int(y1 + (y + bh / 2) * (y2 - y1))

                if overlaps_any((rx1, ry1, rx2, ry2), humans):
                    continue

                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{SKIN_NAMES[cid]} {conf:.2f}",
                    (rx1, ry1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

    # ---- FPS ----
    now = time.time()
    fps = 1 / (now - fps_t + 1e-6)
    fps_t = now

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.imshow("ASF FAST PI", frame)
    if cv2.waitKey(1) == 27:
        break

picam2.stop()
cv2.destroyAllWindows()
