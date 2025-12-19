import cv2
import numpy as np
import math, time, csv
from collections import deque

# ====== USER SETTINGS ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# Real balloon diameter (meters) and calibrated focal length in pixels
D_REAL_M   = 0.30        # e.g., 30 cm
FOCAL_PIX  = 950.0       # set after quick calibration (see notes below)

# HSV range for your balloon color (example: blue)
HSV_LO = (100, 120, 80)
HSV_HI = (135, 255, 255)

# History / smoothing / output
HIST_LEN     = 20         # how many recent samples to keep
ARROW_PIX    = 80         # length of drawn direction arrow
CSV_PATH     = None       # e.g., "balloon_log.csv" to save, or None to disable
MIN_AREA     = 250        # reject tiny blobs

TEXT_COLOR   = (0, 0, 255)  # BGR: red. Use (0,0,0) for black

# ====== HELPERS ======
def largest_blob(mask, min_area=MIN_AREA):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts: return None, 0, (None, None)
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    cx = int(M["m10"]/M["m00"]) if M["m00"] else int(x)
    cy = int(M["m01"]/M["m00"]) if M["m00"] else int(y)
    return c, int(2*r), (cx, cy)   # contour, pixel_diameter, centroid

def z_from_dpix(d_pix, D_real, f_pix):
    if d_pix <= 1: return None
    return (f_pix * D_real) / float(d_pix)

def unit(vx, vy):
    n = math.hypot(vx, vy)
    return (0.0, 0.0) if n < 1e-6 else (vx/n, vy/n)

# ====== INIT ======
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

hist = deque(maxlen=HIST_LEN)  # entries: (t, x, y, z, d_pix)

writer = None
if CSV_PATH:
    f = open(CSV_PATH, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["t", "x_pix", "y_pix", "z_m", "d_pix"])

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break

    # Color segmentation (robust to illumination changes)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)

    c, d_pix, (cx, cy) = largest_blob(mask)
    if c is not None:
        # Estimate distance (Z) from apparent size
        z_m = z_from_dpix(d_pix, D_REAL_M, FOCAL_PIX)

        # Draw
        cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        if z_m is not None:
            cv2.putText(frame, f"Z={z_m:.2f} m", (cx+8, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

        # Update history
        t = time.time()
        hist.append((t, cx, cy, z_m, d_pix))
        if writer: writer.writerow([t, cx, cy, z_m if z_m is not None else "", d_pix])

        # Direction from motion (use a short baseline, e.g., 5 samples back)
        if len(hist) >= 5:
            _, x0, y0, _, _ = hist[-5]
            vx, vy = cx - x0, cy - y0
            ux, uy = unit(vx, vy)
            end = (int(cx + ARROW_PIX*ux), int(cy + ARROW_PIX*uy))
            cv2.arrowedLine(frame, (cx, cy), end, (0, 255, 255), 3, tipLength=0.25)

    cv2.imshow("Balloon tracking (position, scale, direction)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if writer: f.close()
cap.release()
cv2.destroyAllWindows()
