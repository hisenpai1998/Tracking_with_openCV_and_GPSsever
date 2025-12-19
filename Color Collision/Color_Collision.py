import time
try:
    import cv2
except ImportError:
    cv2 = None

# ==== CONFIGURATION ====
COLORS      = ("red", "blue")  
MIN_AREA    = 800              
BLUR_KSIZE  = (5, 5)           
MORPH_K     = (3, 3)           
MARGIN_PX   = 30               

def _color_ranges(color: str):
    c = color.lower()
    if c == "red":
        # Red often needs two HSV ranges due to hue wrap
        return [((0,120,70),(10,255,255)), ((170,120,70),(180,255,255))]
    if c == "blue":
        return [((100,150,50),(140,255,255))]
    if c == "green":
        return [((40,70,50),(80,255,255))]
    return [((0,0,0),(0,0,0))]

def _largest_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)

def _overlap(b1, b2) -> bool:
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    return (x1 < x2+w2 and x1+w1 > x2 and
            y1 < y2+h2 and y1+h1 > y2)

def _edge_distance_px(b1, b2) -> int:
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    # X-axis 
    if x1 > x2 + w2:
        dx = x1 - (x2 + w2)
    elif x2 > x1 + w1:
        dx = x2 - (x1 + w1)
    else:
        dx = 0
    # Y-axis 
    if y1 > y2 + h2:
        dy = y1 - (y2 + h2)
    elif y2 > y1 + h1:
        dy = y2 - (y1 + h1)
    else:
        dy = 0
    return int((dx**2 + dy**2) ** 0.5)

class MotionDetector:
    def __init__(self, source=0):
        if cv2 is None:
            raise RuntimeError("OpenCV not installed. Run: pip install opencv-python")
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera/video. Use --video <file> or run on host with a camera.")
        self.colors = COLORS
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_K)

    def _mask_for_color(self, hsv, color):
        mask = None
        for lo, hi in _color_ranges(color):
            lo, hi = tuple(lo), tuple(hi)
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        # Basic noise reduction & dilation
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel, iterations=1)
        return mask

    def read_motion(self, show=False):
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.02)
            return {"warning": False, "lights": False}

        frame_blur = cv2.GaussianBlur(frame, BLUR_KSIZE, 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        boxes = []
        for color in self.colors:
            mask = self._mask_for_color(hsv, color)
            box = _largest_box(mask)
            boxes.append((color, box))

        present = [b for _, b in boxes if b is not None]
        collision = False
        prewarn   = False

        if len(present) >= 2:
            b1, b2 = present[0], present[1]
            if _overlap(b1, b2):
                collision = True
            else:
                dist = _edge_distance_px(b1, b2)
                prewarn = dist < MARGIN_PX

        if show:
            vis = frame.copy()
            for (color, box) in boxes:
                if box:
                    x,y,w,h = box
                    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(vis, color, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if collision:
                cv2.putText(vis, "COLLISION!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            elif prewarn:
                cv2.putText(vis, "NEAR-COLLISION", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
            cv2.imshow("ColorCollision (with margin)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                self.release()

        return {"warning": (collision or prewarn), "lights": collision}

    def release(self):
        try:
            self.cap.release()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="Webcam index (0,1,...) or video file path")
    ap.add_argument("--show", action="store_true", help="Show GUI window")
    ap.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = infinite)")
    args = ap.parse_args()

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    md = MotionDetector(source=src)
    count = 0
    try:
        while True:
            state = md.read_motion(show=args.show)
            print(f"[FRAME {count}] warning={state['warning']} lights={state['lights']}")
            count += 1
            if args.frames and count >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        md.release()
        print("Done.")
    
