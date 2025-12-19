#!/usr/bin/env python3
import os, time, threading, queue, math
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List

import numpy as np
import cv2

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

# ================== DEFAULTS ==================
STREAM = 0
FRAME_W, FRAME_H = 640, 480

D_REAL_M = 0.30
FOCAL_PIX = 950.0
CALIB_FILE = "calib.npz"

COLOR_PRESETS = {
    "blue":  ((100, 120, 80), (135, 255, 255)),
    "green": ((40,  60, 60),  (85,  255, 255)),
    "red1":  ((0,   120, 80), (10,  255, 255)),
    "red2":  ((170, 120, 80), (180, 255, 255)),
}

# ================== DATA TYPES ==================
@dataclass
class Target3D:
    id: int
    x: float
    y_up: float
    z: float
    u: int
    v: int
    d_pix: float
    t: float

# ================== CALIBRATION + GEOMETRY ==================
def load_intrinsics(calib_path: str, frame_size: Tuple[int,int]):
    FX = FY = FOCAL_PIX
    CX = frame_size[0] / 2.0
    CY = frame_size[1] / 2.0
    map1 = map2 = None
    if os.path.exists(calib_path):
        try:
            data = np.load(calib_path)
            K = data["K"]
            dist = data["dist"]
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, frame_size, 1.0)
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, frame_size, cv2.CV_16SC2)
            FX, FY = float(newK[0,0]), float(newK[1,1])
            CX, CY = float(newK[0,2]), float(newK[1,2])
            print(f"[CALIB] Loaded: fx={FX:.1f} fy={FY:.1f} cx={CX:.1f} cy={CY:.1f}")
        except Exception as e:
            print(f"[CALIB] Failed to load {calib_path}: {e}. Using fallback.")
    else:
        print("[CALIB] No calib.npz found. Using fallback focal/principal.")
    return FX, FY, CX, CY, map1, map2

def depth_from_dpix(d_pix: float, D_real: float, f_pix: float) -> Optional[float]:
    if d_pix <= 1:
        return None
    return (f_pix * D_real) / float(d_pix)

def pix_to_cam(u: float, v: float, z: float, FX: float, FY: float, CX: float, CY: float):
    if z is None:
        return None, None
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY  # image Y down → print Y_up = -y
    return x, y

# ================== DETECTION ==================
def largest_blob(mask: np.ndarray, min_area: int = 250):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        x, y, r = float(x), float(y), float(r)
        cx, cy = int(round(x)), int(round(y))
        d_pix = max(2.0*r, 1.0)
        x0, y0, w, h = cv2.boundingRect(c)
        return dict(cx=cx, cy=cy, d_pix=d_pix, rect=(x0, y0, w, h), contour=c)
    return None

def build_mask(hsv, color: str, lo=None, hi=None):
    if color == "red":
        lo1, hi1 = COLOR_PRESETS["red1"]
        lo2, hi2 = COLOR_PRESETS["red2"]
        m1 = cv2.inRange(hsv, lo1, hi1)
        m2 = cv2.inRange(hsv, lo2, hi2)
        mask = cv2.bitwise_or(m1, m2)
    elif color in COLOR_PRESETS and lo is None and hi is None:
        lo, hi = COLOR_PRESETS[color]
        mask = cv2.inRange(hsv, lo, hi)
    else:
        mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return mask

def open_capture(stream, width=FRAME_W, height=FRAME_H):
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
    if isinstance(stream, str) and not str(stream).isdigit():
        cap = cv2.VideoCapture(stream)
    else:
        try:
            index = int(stream)
        except Exception:
            index = 0
        # Prefer DirectShow on Windows if available
        if os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
    return cap

# ================== OPEN3D VIEWER ==================
class Open3DArrowView:
    def __init__(self, arrow_scale=0.4):
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self._T_prev = np.eye(4)
        self.arrow_scale = float(arrow_scale)

    def start(self):
        if not HAS_O3D or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def update(self, pos_xyz, vel_dir):
        if not self.running:
            return
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait((pos_xyz, vel_dir))
        except queue.Full:
            pass

    def _make_grid(self, extent=3.0, step=0.5):
        pts = []
        lines = []
        xs = np.arange(-extent, extent + 1e-6, step)
        zs = np.arange(-extent, extent + 1e-6, step)
        for x in xs:
            pts.append([x, 0.0, -extent]); pts.append([x, 0.0, extent]); lines.append([len(pts)-2, len(pts)-1])
        for z in zs:
            pts.append([-extent, 0.0, z]); pts.append([extent, 0.0, z]); lines.append([len(pts)-2, len(pts)-1])
        pts = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
        lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.25,0.25,0.25]]), (len(lines),1)))
        ls = o3d.geometry.LineSet(points=pts, lines=lines); ls.colors = colors
        return ls

    def _make_arrow(self):
        a = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02*self.arrow_scale,
            cone_radius=0.04*self.arrow_scale,
            cylinder_height=0.7*self.arrow_scale,
            cone_height=0.25*self.arrow_scale
        )
        a.compute_vertex_normals()
        a.paint_uniform_color([1.0, 0.95, 0.2])
        return a

    def _pose_from_pos_dir(self, pos, vdir):
        T = np.eye(4)
        if pos is not None:
            T[:3,3] = np.array(pos, dtype=np.float64)
        if vdir is None:
            R = np.eye(3)
        else:
            z = np.array(vdir, dtype=np.float64); n = np.linalg.norm(z)
            if n < 1e-6: R = np.eye(3)
            else:
                z = z / n
                up = np.array([0.0, 1.0, 0.0])
                x = np.cross(up, z)
                if np.linalg.norm(x) < 1e-6:
                    up = np.array([1.0, 0.0, 0.0]); x = np.cross(up, z)
                x = x / np.linalg.norm(x)
                y = np.cross(z, x)
                R = np.stack([x, y, z], axis=1)
        T[:3,:3] = R
        return T

    def _run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window("Open3D 3D Direction", width=800, height=600, visible=True)
        grid = self._make_grid(); axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4); arrow = self._make_arrow()
        vis.add_geometry(grid); vis.add_geometry(axes); vis.add_geometry(arrow)
        ctr = vis.get_view_control()
        ctr.set_front([0.0, -0.5, -1.0]); ctr.set_lookat([0.0, 0.0, 0.0]); ctr.set_up([0.0, 1.0, 0.0]); ctr.set_zoom(0.8)

        while self.running and vis.poll_events():
            pos, vdir = None, None
            try:
                while True: pos, vdir = self.q.get_nowait()
            except queue.Empty:
                pass
            if pos is not None:
                try: arrow.transform(np.linalg.inv(self._T_prev))
                except Exception: pass
                self._T_prev = self._pose_from_pos_dir(pos, vdir)
                arrow.transform(self._T_prev)
                vis.update_geometry(arrow)
            vis.update_renderer(); time.sleep(0.03)
        vis.destroy_window()

# ================== DIRECTION SMOOTHING ==================
from collections import deque
def compute_dir_from_hist(hist: deque, min_dt=0.05):
    if len(hist) < 3: return None
    t0, x0, y0, z0 = hist[0]; t1, x1, y1, z1 = hist[-1]
    dt = t1 - t0
    if dt < min_dt: return None
    dv = np.array([x1 - x0, y1 - y0, z1 - z0], dtype=np.float64)
    n = np.linalg.norm(dv)
    if n < 1e-6: return None
    return (dv / n).tolist()

# ================== WORKER NODE ==================
class TrackingNode:
    def __init__(self,
                 stream=STREAM,
                 color="blue",
                 hsv_range: Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = None,
                 diameter_m: float = D_REAL_M,
                 calib_path: str = CALIB_FILE,
                 print_hz: float = 5.0,
                 show_window: bool = False,
                 show_o3d: bool = False):
        self.stream = stream
        self.color = color
        self.hsv_range = hsv_range
        self.diameter_m = diameter_m
        self.calib_path = calib_path
        self.print_hz = float(print_hz)
        self.show_window = bool(show_window)
        self.show_o3d = bool(show_o3d) and HAS_O3D

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest: Optional[Target3D] = None
        self._callback: Optional[Callable[[List[Target3D]], None]] = None

        self._FX = self._FY = FOCAL_PIX
        self._CX = FRAME_W/2.0; self._CY = FRAME_H/2.0
        self._map1 = self._map2 = None

        self._pos_hist = deque(maxlen=10)
        self._last_dir = [0.0, 0.0, 1.0]
        self._o3d_view = Open3DArrowView() if self.show_o3d else None

    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="TrackingNode", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2.0)
        self._thread = None

    def set_callback(self, cb: Optional[Callable[[List[Target3D]], None]]):
        with self._lock:
            self._callback = cb

    def get_latest(self) -> Optional[Target3D]:
        with self._lock:
            return self._latest

    def get_all(self) -> List[Target3D]:
        with self._lock:
            return [self._latest] if self._latest is not None else []

    def _run(self):
        cap = open_capture(self.stream, FRAME_W, FRAME_H)
        if not cap or not cap.isOpened():
            print("[ERROR] Could not open video stream."); return

        ok, frame = cap.read()
        if not ok: print("[ERROR] Failed to read first frame."); cap.release(); return
        h, w = frame.shape[:2]
        self._FX, self._FY, self._CX, self._CY, self._map1, self._map2 = load_intrinsics(self.calib_path, (w, h))
        last_print = 0.0; print_period = 1.0 / max(1e-3, self.print_hz)

        if self._o3d_view: self._o3d_view.start()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok: break
            if self._map1 is not None:
                frame = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.hsv_range:
                lo, hi = self.hsv_range
                mask = build_mask(hsv, color="custom", lo=lo, hi=hi)
            else:
                mask = build_mask(hsv, color=self.color)

            blob = largest_blob(mask, min_area=250)
            tgt = None

            if blob:
                cx, cy, d_pix = blob["cx"], blob["cy"], float(blob["d_pix"])
                z = depth_from_dpix(d_pix, self.diameter_m, self._FX)
                x, y_img = pix_to_cam(cx, cy, z, self._FX, self._FY, self._CX, self._CY)
                y_up = -y_img if y_img is not None else None

                if z is not None and x is not None and y_up is not None:
                    now = time.time()
                    tgt = Target3D(id=0, x=float(x), y_up=float(y_up), z=float(z),
                                   u=int(cx), v=int(cy), d_pix=d_pix, t=now)
                    # Update history and 3D arrow
                    self._pos_hist.append((now, tgt.x, tgt.y_up, tgt.z))
                    vel_dir = compute_dir_from_hist(self._pos_hist)
                    if vel_dir is not None: self._last_dir = vel_dir
                    if self._o3d_view: self._o3d_view.update((tgt.x, tgt.y_up, tgt.z), self._last_dir)

                    if self.show_window:
                        x0, y0, w0, h0 = blob["rect"]
                        cv2.rectangle(frame, (x0, y0), (x0+w0, y0+h0), (0, 255, 255), 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                        cv2.putText(frame, f"X={tgt.x:.2f}m Y={tgt.y_up:.2f}m Z={tgt.z:.2f}m",
                                    (x0, max(0, y0-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

                    if (now - last_print) >= print_period:
                        print(f"X={tgt.x:.3f} m  Y={tgt.y_up:.3f} m  Z={tgt.z:.3f} m   (u={tgt.u}, v={tgt.v}, d_pix={tgt.d_pix:.1f})")
                        last_print = now

            # Publish latest and call callback
            with self._lock:
                self._latest = tgt
                cb = self._callback

            if cb is not None and tgt is not None:
                try:
                    cb([tgt])
                except Exception as e:
                    # Keep the loop alive even if client callback throws
                    print(f"[WARN] callback error: {e}")

            if self.show_window:
                cv2.imshow("3D Detection Tracker (API)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

        cap.release()
        if self.show_window: cv2.destroyAllWindows()
        if self._o3d_view: self._o3d_view.stop()

# ================== PUBLIC API ==================
class TrackingAPI:
    """
    API for 3D detection/tracking (runs in its own thread).
    Usage:
        api = TrackingAPI(stream=0, color='blue', show_window=False, show_o3d=False)
        api.start()
        pos = api.getPosition()          # Target3D or None
        api.setPositionCallback(lambda rows: print(rows))
        api.stop()
    """
    def __init__(self,
                 stream=STREAM,
                 msg_type: str = 'single',  # reserved for future multi-target
                 color: str = 'blue',
                 hsv_range: Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = None,
                 diameter_m: float = D_REAL_M,
                 calib_path: str = CALIB_FILE,
                 print_hz: float = 5.0,
                 show_window: bool = False,
                 show_o3d: bool = False):
        self.node = TrackingNode(stream=stream, color=color, hsv_range=hsv_range,
                                 diameter_m=diameter_m, calib_path=calib_path,
                                 print_hz=print_hz, show_window=show_window, show_o3d=show_o3d)

    def start(self):
        self.node.start()

    def stop(self):
        self.node.stop()

    def getPosition(self, targetID=None):
        """
        - None: returns latest Target3D or None
        - int:   currently only id=0 is produced; returns latest or None
        - list[int]: returns list with latest if id present (future expansion)
        """
        latest = self.node.get_latest()
        if targetID is None:
            return latest
        if isinstance(targetID, list):
            return [latest] if latest is not None and latest.id in targetID else []
        return latest if (latest is not None and latest.id == int(targetID)) else None

    def getPositions(self) -> List[Target3D]:
        return self.node.get_all()

    def setPositionCallback(self, callback: Optional[Callable[[List[Target3D]], None]]):
        self.node.set_callback(callback)

    # Context manager convenience
    def __enter__(self):
        self.start(); return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Tiny CLI for quick manual run (optional)
if __name__ == "__main__":
    api = TrackingAPI(stream=STREAM, color='blue', show_window=True, show_o3d=True)
    api.setPositionCallback(lambda rows: rows and print(f"[CB] {rows[0]}"))
    api.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        api.stop()
