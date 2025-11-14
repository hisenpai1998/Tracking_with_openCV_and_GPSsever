import argparse
import json
import socket
import time
from typing import Dict, Tuple, Optional

import numpy as np
import math

try:
    import cv2
except ImportError:
    cv2 = None


def ensure_opencv():
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. Run: pip install opencv-python")


def send_state(sock: socket.socket, target: Tuple[str, int], rid: str, x: float, y: float, theta=None):
    msg = {"id": str(rid), "x": float(x), "y": float(y)}
    if theta is not None:
        msg["theta"] = float(theta)
    data = json.dumps(msg).encode("utf-8")
    sock.sendto(data, target)


def _dir_from_motion(dx: float, dy: float, prev: Optional[str], threshold: float = 2.0) -> str:
    """
    Map motion vector to one of {E,S,W,N}. If speed below threshold, keep previous if provided.
    Image coords: x right (E), y down (S), so -y is North.
    """
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < threshold:
        return prev or "?"
    # Choose dominant axis
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    else:
        return "S" if dy > 0 else "N"


def _overlap(b1, b2) -> bool:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)


def _edge_distance_px(b1, b2) -> int:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    if x1 > x2 + w2:
        dx = x1 - (x2 + w2)
    elif x2 > x1 + w1:
        dx = x2 - (x1 + w1)
    else:
        dx = 0
    if y1 > y2 + h2:
        dy = y1 - (y2 + h2)
    elif y2 > y1 + h1:
        dy = y2 - (y1 + h1)
    else:
        dy = 0
    return int((dx * dx + dy * dy) ** 0.5)


def run_aruco(args):
    ensure_opencv()
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera/video. Use --source <index|path>.")

    # Choose dictionary
    DICT_MAP = {
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_50": cv2.aruco.DICT_5X5_50,
        "6x6_50": cv2.aruco.DICT_6X6_50,
        "apriltag_36h11": getattr(cv2.aruco, "DICT_APRILTAG_36h11", cv2.aruco.DICT_4X4_50),
    }
    dict_id = DICT_MAP.get(args.aruco_dict.lower(), cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # Detector params and (new vs old) API handling
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    use_new = hasattr(cv2.aruco, "ArucoDetector")
    detector = cv2.aruco.ArucoDetector(dictionary, parameters) if use_new else None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.broadcast:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    target = (args.target_ip, args.target_port)

    last = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if use_new:
                corners, ids, _rej = detector.detectMarkers(gray)
            else:
                corners, ids, _rej = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

            if ids is not None and len(ids) > 0:
                ids = ids.flatten()
                for i, rid in enumerate(ids):
                    c = corners[i]
                    c = c.reshape(-1, 2)  # (4,2)
                    cx, cy = np.mean(c, axis=0)
                    # orientation from first edge (corner 0 -> 1)
                    v = c[1] - c[0]
                    theta = float(np.arctan2(v[1], v[0]))
                    send_state(sock, target, str(rid), cx, cy, theta)

            if args.show:
                disp = frame.copy()
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(disp, corners, ids)
                cv2.imshow("GPS Py Server - ArUco", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            # FPS limit
            if args.fps_limit > 0:
                dt = time.time() - last
                need = max(0.0, (1.0 / args.fps_limit) - dt)
                if need > 0:
                    time.sleep(need)
                last = time.time()
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def parse_id_color_map(s: str) -> Dict[str, str]:
    # format: "R1:red,R2:blue"
    out: Dict[str, str] = {}
    if not s:
        return out
    items = s.split(',')
    for it in items:
        if ':' in it:
            rid, col = it.split(':', 1)
            out[rid.strip()] = col.strip()
    return out


def _color_ranges(color: str, green_override: Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = None):
    c = color.lower()
    if c == "red":
        return [((0,120,70),(10,255,255)), ((170,120,70),(180,255,255))]
    if c == "blue":
        return [((100,150,50),(140,255,255))]
    if c == "green":
        if green_override is not None:
            lo, hi = green_override
            return [(tuple(lo), tuple(hi))]
        # Broadened default based on typical indoor lighting
        return [((35,40,40),(90,255,255))]
    return [((0,0,0),(0,0,0))]


def _largest_box(mask, min_area=800):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)


def _clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _largest_good_contour(mask, min_area=400):
    """Pick a compact, relatively round, non-spiky contour (used for spiral blob)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        p = cv2.arcLength(c, True) + 1e-6
        circularity = 4 * math.pi * area / (p * p)  # ~1 for circle-like
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area  # ~1 for filled shapes
        score = 0.6 * circularity + 0.4 * solidity
        if score > best_score:
            best_score = score
            best = c
    return best


def _centroid_of(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def _find_all_good_contours(mask, min_area=400):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    picks = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        p = cv2.arcLength(c, True) + 1e-6
        circularity = 4 * math.pi * area / (p * p)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        score = 0.6 * circularity + 0.4 * solidity
        picks.append((score, area, c))
    picks.sort(key=lambda t: (-t[0], -t[1]))
    return [c for _s, _a, c in picks]


def run_color(args):
    ensure_opencv()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera/video.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.broadcast:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    target = (args.target_ip, args.target_port)

    id_map = parse_id_color_map(args.ids)
    if not id_map:
        # default mapping
        id_map = {"R1": "red", "R2": "green"}

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Track previous centers and last cardinal direction per id
    prev_centers: Dict[str, Tuple[float, float]] = {}
    last_dir: Dict[str, str] = {}
    last = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)

            # Collect boxes for collision warning
            bboxes: Dict[str, Tuple[int, int, int, int]] = {}
            # Optional green HSV override from CLI
            green_lo_hi = None
            if args.green_hsv:
                try:
                    parts = [int(p) for p in args.green_hsv.split(',')]
                    if len(parts) == 6:
                        green_lo_hi = ((parts[0], parts[1], parts[2]), (parts[3], parts[4], parts[5]))
                except Exception:
                    green_lo_hi = None

            for rid, color in id_map.items():
                mask = None
                for lo, hi in _color_ranges(color, green_override=green_lo_hi):
                    lo, hi = tuple(lo), tuple(hi)
                    m = cv2.inRange(hsv, lo, hi)
                    mask = m if mask is None else cv2.bitwise_or(mask, m)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
                box = _largest_box(mask, min_area=args.min_area)
                if box:
                    x, y, w, h = box
                    cx, cy = x + w / 2, y + h / 2
                    send_state(sock, target, rid, cx, cy, theta=None)
                    bboxes[rid] = (x, y, w, h)
                    # Determine motion-based cardinal direction
                    dx = cx - prev_centers.get(rid, (cx, cy))[0]
                    dy = cy - prev_centers.get(rid, (cx, cy))[1]
                    d = _dir_from_motion(dx, dy, last_dir.get(rid), threshold=args.dir_threshold)
                    prev_centers[rid] = (cx, cy)
                    last_dir[rid] = d
                if args.show:
                    if box:
                        # Draw bbox and annotation with ID, color, position and direction
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                        label = f"{rid} ({color})"
                        cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                        cv2.putText(frame, f"pos=({int(cx)},{int(cy)}) dir={last_dir.get(rid,'?')}", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if args.show:
                # Collision/near-collision warning across detected boxes
                collision = False
                prewarn = False
                keys = list(bboxes.keys())
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        b1 = bboxes[keys[i]]
                        b2 = bboxes[keys[j]]
                        if _overlap(b1, b2):
                            collision = True
                        else:
                            dist = _edge_distance_px(b1, b2)
                            if dist < args.warn_margin:
                                prewarn = True
                if collision:
                    cv2.putText(frame, "COLLISION!", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                elif prewarn:
                    cv2.putText(frame, "WARNING: NEAR-COLLISION", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                cv2.imshow("GPS Py Server - Color", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if args.fps_limit > 0:
                dt = time.time() - last
                need = max(0.0, (1.0 / args.fps_limit) - dt)
                if need > 0:
                    time.sleep(need)
                last = time.time()
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def run_spiral(args):
    """Detect spiral-like blobs and track multiple objects with stable IDs; broadcast each object's centroid."""
    ensure_opencv()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera/video.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.broadcast:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    target = (args.target_ip, args.target_port)

    last = time.time()
    # Simple nearest-neighbor tracker state
    tracks: Dict[int, Dict[str, float]] = {}  # id -> {x,y,missed}
    next_id = 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Adaptive grayscale segmentation (robust to illumination)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = _clahe_gray(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 21, 5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)

            contours = _find_all_good_contours(mask, min_area=args.spiral_min_area)
            detections = []  # list of (cx, cy, area, contour)
            for c in contours:
                ctr = _centroid_of(c)
                if ctr is not None:
                    detections.append((float(ctr[0]), float(ctr[1]), cv2.contourArea(c), c))

            # Associate detections to existing tracks (greedy nearest-neighbor)
            assigned_tracks = set()
            assigned_dets = set()
            # Precompute distances
            for tid, state in list(tracks.items()):
                state['missed'] = state.get('missed', 0) + 1
            if detections and tracks:
                dist_list = []
                for di, (dx, dy, _da, _dc) in enumerate(detections):
                    for tid, state in tracks.items():
                        d2 = (state['x'] - dx) ** 2 + (state['y'] - dy) ** 2
                        dist_list.append((d2, di, tid))
                dist_list.sort(key=lambda t: t[0])
                for d2, di, tid in dist_list:
                    if di in assigned_dets or tid in assigned_tracks:
                        continue
                    if d2 <= (args.spiral_match_max_dist ** 2):
                        # associate
                        tracks[tid]['x'] = detections[di][0]
                        tracks[tid]['y'] = detections[di][1]
                        tracks[tid]['missed'] = 0
                        assigned_dets.add(di)
                        assigned_tracks.add(tid)

            # Create new tracks for unassigned detections
            for di, (dx, dy, _da, _dc) in enumerate(detections):
                if di in assigned_dets:
                    continue
                tracks[next_id] = {'x': dx, 'y': dy, 'missed': 0}
                assigned_tracks.add(next_id)
                next_id += 1

            # Remove stale tracks
            to_del = [tid for tid, st in tracks.items() if st.get('missed', 0) > args.spiral_max_missed]
            for tid in to_del:
                del tracks[tid]

            # Send UDP for each active track
            for tid, st in tracks.items():
                send_state(sock, target, str(tid), st['x'], st['y'], theta=None)

            if args.show:
                disp = frame.copy()
                # Draw all detections and track labels
                for _dx, _dy, _da, c in detections:
                    cv2.drawContours(disp, [c], -1, (0, 200, 50), 2)
                for tid, st in tracks.items():
                    cx, cy = int(st['x']), int(st['y'])
                    cv2.circle(disp, (cx, cy), 6, (0, 0, 255), 2)
                    cv2.putText(disp, f"ID={tid} pos=({cx},{cy})", (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 255), 2)
                cv2.imshow("GPS Py Server - Spiral", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if args.fps_limit > 0:
                dt = time.time() - last
                need = max(0.0, (1.0 / args.fps_limit) - dt)
                if need > 0:
                    time.sleep(need)
                last = time.time()
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Python GPS server: detect markers/spirals and broadcast positions over UDP JSON for robot_nav_demo.py")
    ap.add_argument("mode", choices=["aruco", "color", "spiral"], help="Detection source")
    ap.add_argument("--source", default="0", help="Camera index (0,1,...) or video path")
    ap.add_argument("--target-ip", default="127.0.0.1", help="UDP destination IP (robot_nav_demo listener host)")
    ap.add_argument("--target-port", type=int, default=5005, help="UDP destination port")
    ap.add_argument("--broadcast", action="store_true", help="Enable UDP broadcast (sets SO_BROADCAST)")
    ap.add_argument("--fps-limit", type=float, default=30.0, help="Limit processing FPS (0 = unlimited)")
    ap.add_argument("--show", action="store_true", help="Show detection visualization")

    # ArUco options
    ap.add_argument("--aruco-dict", default="4x4_50", help="Dictionary: 4x4_50, 4x4_100, 5x5_50, 6x6_50, apriltag_36h11")

    # Color options
    ap.add_argument("--ids", default="R1:red,R2:green", help="Map of id:color, e.g. 'R1:red,R2:green'")
    ap.add_argument("--min-area", type=int, default=800, help="Min area for color blobs")
    ap.add_argument("--green-hsv", default="", help="Override green HSV lo,hi as 'Hlo,Slo,Vlo,Hhi,Shi,Vhi', e.g. '35,40,40,90,255,255'")
    ap.add_argument("--warn-margin", type=int, default=30, help="Near-collision margin in pixels")
    ap.add_argument("--dir-threshold", type=float, default=2.0, help="Motion threshold (pixels/frame) to update E/S/W/N direction")

    # Spiral options
    ap.add_argument("--spiral-min-area", type=int, default=400, help="Min area for spiral contour")
    ap.add_argument("--spiral-match-max-dist", type=float, default=60.0, help="Max distance (pixels) to associate detections with existing tracks")
    ap.add_argument("--spiral-max-missed", type=int, default=8, help="Max frames a track can be missed before removal")

    args = ap.parse_args()

    # Normalize source
    try:
        args.source = int(args.source)
    except ValueError:
        pass

    if args.mode == "aruco":
        run_aruco(args)
    elif args.mode == "color":
        run_color(args)
    elif args.mode == "spiral":
        run_spiral(args)


if __name__ == "__main__":
    main()
