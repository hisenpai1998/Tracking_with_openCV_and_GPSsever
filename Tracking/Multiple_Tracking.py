import cv2
import numpy as np
import math, time, csv
import os
import argparse
from urllib.parse import urlparse, urlunparse
from collections import deque

# ================== USER SETTINGS ==================
STREAM = 0                       # CAM index or IP URL (e.g., "rtsp://..." or "http://...")
FRAME_W, FRAME_H = 640, 480

# Camera intrinsics (will be loaded from calib.npz if present)
CALIB_FILE = "calib.npz"  # contains K (3x3) and dist (1x5 or 1x8)

# Balloon real diameter (meters) and calibrated focal length in pixels
D_REAL_M  = 0.30                 # e.g., 30 cm
FOCAL_PIX = 950.0                # fallback if no calib file

# HSV range (example: BLUE). For RED, use two ranges and OR them.
HSV_LO = (100, 120, 80)
HSV_HI = (135, 255, 255)

MIN_AREA     = 250               # ignore tiny blobs
ARROW_PIX    = 80                # direction arrow length
TEXT_COLOR   = (0, 0, 255)       # BGR
DRAW_COLOR   = (255, 0, 0)       # contour color
CENTER_COLOR = (0, 255, 0)

# 3D direction arrow settings
DRAW_3D_ARROW = True            # draw 3D motion arrow using metric coords and projection
ARROW_LEN_M   = 0.40            # 3D arrow length in meters
ARROW_3D_COLOR = (0, 255, 255)  # yellow-ish

# Multi-object tracking
MAX_TRACKS         = 50
ASSOC_DIST_THRESH  = 60.0         # pixels: max distance to re-associate
HIST_LEN           = 20           # history buffer per track
FORGET_SECONDS     = 1.0          # drop track if unseen this long

# Logging (set None to disable)
CSV_PATH = "C:\\Users\\Mezin\\OneDrive - Högskolan i Halmstad\\Skrivbordet\\Pose_log_MT.csv"   # e.g., "multi_balloon_log.csv"

# ================== HELPERS ==================
def find_blobs(mask, min_area=MIN_AREA):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: 
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(x)
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(y)
        d_pix = int(2*r)
        blobs.append({"contour": c, "cx": cx, "cy": cy, "d_pix": d_pix})
    return blobs

def z_from_dpix(d_pix, D_real, f_pix):
    if d_pix <= 1: return None
    return (f_pix * D_real) / float(d_pix)

def unit(vx, vy):
    n = math.hypot(vx, vy)
    return (0.0, 0.0) if n < 1e-6 else (vx/n, vy/n)

# Simple nearest-neighbor data association
class Track:
    def __init__(self, tid, cx, cy, d_pix, z_m, t, hist_len):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.d_pix = d_pix
        self.z_m = z_m
        self.last_t = t
        self.history = deque(maxlen=hist_len)  # (t, x, y, z, d)

    def update(self, cx, cy, d_pix, z_m, t):
        self.cx, self.cy = cx, cy
        self.d_pix = d_pix
        self.z_m = z_m
        self.last_t = t
        self.history.append((t, cx, cy, z_m, d_pix))

def associate_tracks(tracks, blobs, t, dist_thresh):
    """
    Greedy nearest neighbor: match each blob to closest track under threshold.
    Unmatched blobs -> new tracks. Unseen tracks are kept; caller can prune by time.
    """
    used_tracks = set()
    used_blobs = set()
    # Precompute distances
    dists = []
    for bi, b in enumerate(blobs):
        for ti, tr in enumerate(tracks):
            d = math.hypot(b["cx"] - tr.cx, b["cy"] - tr.cy)
            dists.append((d, ti, bi))
    dists.sort(key=lambda x: x[0])

    # Assign greedily
    for d, ti, bi in dists:
        if d > dist_thresh: 
            continue
        if ti in used_tracks or bi in used_blobs:
            continue
        tr = tracks[ti]
        b  = blobs[bi]
        z_m = z_from_dpix(b["d_pix"], D_REAL_M, FOCAL_PIX)
        tr.update(b["cx"], b["cy"], b["d_pix"], z_m, t)
        used_tracks.add(ti)
        used_blobs.add(bi)

    # Create new tracks for unmatched blobs
    next_id = (max([tr.id for tr in tracks], default=0) + 1) if tracks else 1
    for bi, b in enumerate(blobs):
        if bi in used_blobs:
            continue
        if len(tracks) >= MAX_TRACKS:
            break
        z_m = z_from_dpix(b["d_pix"], D_REAL_M, FOCAL_PIX)
        tr = Track(next_id, b["cx"], b["cy"], b["d_pix"], z_m, t, HIST_LEN)
        tr.history.append((t, b["cx"], b["cy"], z_m, b["d_pix"]))
        tracks.append(tr)
        next_id += 1

# ================== MAIN ==================
# Try to load camera intrinsics
K = None
dist = None
FX = FY = FOCAL_PIX
CX = FRAME_W / 2.0
CY = FRAME_H / 2.0
if os.path.exists(CALIB_FILE):
    try:
        data = np.load(CALIB_FILE)
        K = data["K"]
        dist = data["dist"]
        FX, FY = float(K[0,0]), float(K[1,1])
        CX, CY = float(K[0,2]), float(K[1,2])
        print(f"[CALIB] Loaded: fx={FX:.1f} fy={FY:.1f} cx={CX:.1f} cy={CY:.1f}")
    except Exception as e:
        print(f"[CALIB] Failed to load {CALIB_FILE}: {e}. Using fallback.")

def depth_from_dpix(d_pix):
    # Use FX for depth scaling (FY would be similar). Requires known D_REAL_M.
    if d_pix <= 1:
        return None
    return (FX * D_REAL_M) / float(d_pix)

def pix_to_cam(u, v, z):
    # Convert pixel (u,v) to camera coordinates in meters at depth z
    # X right, Y down, Z forward. Negate Y if you want "up" positive.
    if z is None:
        return (None, None)
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return (x, y)

def cam_to_pix(x_m, y_m, z_m):
    # Project camera coordinates (meters) to pixel using intrinsics
    if z_m is None or z_m <= 1e-6:
        return None
    u = FX * x_m / z_m + CX
    v = FY * y_m / z_m + CY
    return (int(round(u)), int(round(v)))

def _with_auth(url, user=None, password=None):
    try:
        if not (user and password):
            return url
        p = urlparse(url)
        # Insert user:pass@ into netloc
        netloc = p.netloc
        if "@" in netloc:
            return url  # already has auth
        auth_netloc = f"{user}:{password}@{netloc}"
        return urlunparse((p.scheme, auth_netloc, p.path or "", p.params or "", p.query or "", p.fragment or ""))
    except Exception:
        return url

def _derive_candidates_from_viewer(viewer_url, user=None, password=None):
    """Given a camera web viewer URL (like .../view/viewer_index.shtml?id=7),
    return a list of common stream endpoints (HTTP MJPEG and RTSP) to try."""
    candidates = []
    try:
        p = urlparse(viewer_url)
        base = f"{p.scheme}://{p.hostname}"
        if p.port:
            base = f"{base}:{p.port}"

        # Common MJPEG endpoints (Axis, generic, some OEMs)
        mjpeg_paths = [
            "/mjpg/video.mjpg",
            "/axis-cgi/mjpg/video.cgi",
            "/cgi-bin/mjpg/video.cgi",
            "/video.cgi",
            "/video/mjpg.cgi",
            "/stream/video.mjpeg",
            "/mjpeg.cgi",
            "/index.mjpg",
        ]
        for path in mjpeg_paths:
            candidates.append(_with_auth(base + path, user, password))

        # Axis AMP (H.264 over HTTP)
        candidates.append(_with_auth(base + "/axis-media/media.amp", user, password))

        # Snapshot endpoints (not continuous streams; last resort)
        snapshot_paths = [
            "/snapshot.jpg",
            "/jpg/image.jpg",
            "/image.jpg",
        ]
        for path in snapshot_paths:
            candidates.append(_with_auth(base + path, user, password))

        # Common RTSP endpoints (Hikvision, Dahua, generic ONVIF/main)
        host = p.hostname
        auth_prefix = f"{user}:{password}@" if (user and password) else ""
        rtsp_base = f"rtsp://{auth_prefix}{host}:554"
        rtsp_paths = [
            "/Streaming/Channels/101",                  # Hikvision main
            "/Streaming/Channels/102",                  # Hikvision sub
            "/h264/ch1/main/av_stream",                 # Many OEMs
            "/h264/ch1/sub/av_stream",                  # Substream
            "/cam/realmonitor?channel=1&subtype=0",     # Dahua main
            "/cam/realmonitor?channel=1&subtype=1",     # Dahua sub
            "/onvif-media/media.amp",                   # Axis ONVIF media
        ]
        for path in rtsp_paths:
            candidates.append(rtsp_base + path)
    except Exception:
        pass
    return candidates

def open_video_capture(stream, user=None, password=None, frame_w=None, frame_h=None, try_guess=True):
    """Open cv2.VideoCapture for int index or IP URL. If given a viewer page URL,
    try common stream endpoints. Returns (cap, used_url)."""
    used_url = stream
    # If stream is a string but looks like a viewer page, try candidates
    if isinstance(stream, str) and stream.lower().endswith((".shtml", ".html")) and try_guess:
        cands = _derive_candidates_from_viewer(stream, user, password)
        # Also try the provided URL as-is (with auth injected) first
        cands.insert(0, _with_auth(stream, user, password))
        for url in cands:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                # Try to read one frame quickly
                ok, _ = cap.read()
                if ok:
                    print(f"[VIDEO] Connected via: {url}")
                    return cap, url
                cap.release()
        print("[VIDEO] Failed to auto-detect a stream endpoint from viewer page.\n"
              "        If your camera requires credentials, pass --user and --password.\n"
              "        Or provide a direct RTSP/MJPEG URL with --stream.")
        return cv2.VideoCapture(), None

    # If stream is a string (direct URL)
    if isinstance(stream, str):
        url = _with_auth(stream, user, password)
        cap = cv2.VideoCapture(url)
        used_url = url
    else:
        # Assume local webcam index
        cap = cv2.VideoCapture(stream, cv2.CAP_DSHOW)
        if frame_w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
        if frame_h:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    return cap, used_url

# CLI args to override stream and credentials
parser = argparse.ArgumentParser(description="Multi-object color tracking with optional IP camera stream")
parser.add_argument("--stream", "-s", type=str, default=None, help="Camera index (int) or URL (rtsp/http). Example: rtsp://user:pass@192.168.1.2:554/Streaming/Channels/101")
parser.add_argument("--user", type=str, default=None, help="Username for the camera (optional)")
parser.add_argument("--password", type=str, default=None, help="Password for the camera (optional)")
parser.add_argument("--arrow3d", type=str, choices=["on","off"], default=None, help="Enable 3D motion arrow (on/off). Default: on")
parser.add_argument("--arrow3d-len", type=float, default=None, help="3D arrow length in meters (default from code)")
args, unknown = parser.parse_known_args()

if args.stream is not None:
    try:
        # Allow numeric index if provided
        STREAM = int(args.stream)
    except ValueError:
        STREAM = args.stream

# Override 3D arrow settings from CLI if provided
if args.arrow3d is not None:
    DRAW_3D_ARROW = (args.arrow3d == "on")
if args.arrow3d_len is not None:
    ARROW_LEN_M = float(args.arrow3d_len)

# Video open (DirectShow for Windows webcam index or IP URL)
# Improve RTSP reliability (FFmpeg backend) if present
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

# If user provided a base HTTP URL without a specific path, try known Axis/Generic endpoints automatically
cap = None
USED_URL = None
if isinstance(STREAM, str):
    try:
        p = urlparse(STREAM)
        looks_like_base_http = p.scheme in ("http", "https") and (p.path in ("", "/") or p.path is None)
    except Exception:
        looks_like_base_http = False

    if looks_like_base_http:
        # Try candidates derived from host
        cands = _derive_candidates_from_viewer(STREAM, args.user, args.password)
        # Prioritize Axis typical endpoints first
        axis_first = [
            "/axis-cgi/mjpg/video.cgi",
            "/axis-media/media.amp",
            "/mjpg/video.mjpg",
        ]
        # Move prioritized to the front
        prioritized = []
        rest = []
        for url in cands:
            try:
                up = urlparse(url)
                if any(url.endswith(path) for path in axis_first):
                    prioritized.append(url)
                else:
                    rest.append(url)
            except Exception:
                rest.append(url)
        try_list = prioritized + rest

        for url in try_list:
            cap_try = cv2.VideoCapture(url)
            if cap_try.isOpened():
                ok, _ = cap_try.read()
                if ok:
                    cap = cap_try
                    USED_URL = url
                    print(f"[VIDEO] Connected via: {url}")
                    break
                cap_try.release()

if cap is None:
    cap, USED_URL = open_video_capture(STREAM, user=args.user, password=args.password, frame_w=FRAME_W, frame_h=FRAME_H)
if USED_URL:
    print(f"[VIDEO] Using stream: {USED_URL}")
else:
    print(f"[VIDEO] Using stream: {STREAM}")
if not cap or not cap.isOpened():
    print("[ERROR] Could not open video stream. Check URL/index and credentials.")
    raise SystemExit(1)

# If we have intrinsics+distortion, prepare undistort maps
map1 = map2 = None
if K is not None and dist is not None:
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (FRAME_W, FRAME_H), 1.0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (FRAME_W, FRAME_H), cv2.CV_16SC2)
    # Update principal point/focal with the rectified matrix
    FX, FY = float(newK[0,0]), float(newK[1,1])
    CX, CY = float(newK[0,2]), float(newK[1,2])
    print(f"[RECT] Using newK: fx={FX:.1f} fy={FY:.1f} cx={CX:.1f} cy={CY:.1f}")

tracks = []  # list[Track]

writer = None
csv_file = None
if CSV_PATH:
    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["t", "track_id", "u_pix", "v_pix", "x_m", "y_m", "z_m", "d_pix"])

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    if map1 is not None:
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    now = time.time()

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # If RED balloon, combine two ranges (uncomment + set ranges):
    # mask1 = cv2.inRange(hsv, (0,120,80), (10,255,255))
    # mask2 = cv2.inRange(hsv, (170,120,80), (180,255,255))
    # mask = cv2.bitwise_or(mask1, mask2)

    # Single-range example (BLUE):
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)

    blobs = find_blobs(mask, MIN_AREA)

    # Associate/update tracks
    associate_tracks(tracks, blobs, now, ASSOC_DIST_THRESH)

    # Prune stale tracks
    tracks = [tr for tr in tracks if (now - tr.last_t) <= FORGET_SECONDS]

    # Draw and log
    for tr in tracks:
        # Direction arrow from motion over a short baseline
        if len(tr.history) >= 5:
            # 2D arrow (image plane) as fallback for visualization
            _, x0_pix, y0_pix, _, _ = tr.history[-5]
            vx2d, vy2d = tr.cx - x0_pix, tr.cy - y0_pix
            ux2d, uy2d = unit(vx2d, vy2d)
            end2d = (int(tr.cx + ARROW_PIX*ux2d), int(tr.cy + ARROW_PIX*uy2d))
            if not DRAW_3D_ARROW:
                cv2.arrowedLine(frame, (tr.cx, tr.cy), end2d, (0, 255, 255), 3, tipLength=0.25)

        # Contour: draw the closest blob (approximate). For multi-blob we already used nearest neighbor,
        # so just draw a small circle + text
        cv2.circle(frame, (tr.cx, tr.cy), 6, CENTER_COLOR, -1)

        # Compute metric coordinates
        z_m = tr.z_m
        x_m, y_m = pix_to_cam(tr.cx, tr.cy, z_m)

        # 3D motion arrow
        if DRAW_3D_ARROW and len(tr.history) >= 5 and z_m is not None:
            _, x0_pix, y0_pix, z0_m, _ = tr.history[-5]
            if z0_m is not None:
                x0_m, y0_m = pix_to_cam(x0_pix, y0_pix, z0_m)
                if None not in (x0_m, y0_m, x_m, y_m):
                    dx, dy, dz = (x_m - x0_m), (y_m - y0_m), (z_m - z0_m)
                    mag = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if mag > 1e-6:
                        ux3d, uy3d, uz3d = dx/mag, dy/mag, dz/mag
                        end3d = (x_m + ARROW_LEN_M*ux3d, y_m + ARROW_LEN_M*uy3d, z_m + ARROW_LEN_M*uz3d)
                        p0 = cam_to_pix(x_m, y_m, z_m)
                        p1 = cam_to_pix(*end3d)
                        if p0 and p1:
                            cv2.arrowedLine(frame, p0, p1, ARROW_3D_COLOR, 3, tipLength=0.25)

        # Draw ID and XYZ (Y shown with minus to display "up" positive)
        if z_m:
            cv2.putText(frame,
                        f"ID {tr.id}  X={x_m:.2f}m Y={-y_m:.2f}m Z={z_m:.2f}m",
                        (tr.cx+8, tr.cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"ID {tr.id}", (tr.cx+8, tr.cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

        if writer:
            writer.writerow([
                tr.last_t, tr.id, tr.cx, tr.cy,
                f"{x_m:.3f}" if x_m is not None else "",
                f"{y_m:.3f}" if y_m is not None else "",
                f"{z_m:.3f}" if z_m is not None else "",
                tr.d_pix
            ])

    cv2.imshow("Multi-object color tracking (pos, scale->Z, direction)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

if writer:
    csv_file.close()
cap.release()
cv2.destroyAllWindows()
