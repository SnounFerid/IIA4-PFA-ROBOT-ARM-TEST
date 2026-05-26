"""
A4 Paper Camera Calibration Tool  (fixed)
==========================================
Uses an A4 sheet (210 x 297 mm) as a reference to calibrate
pixel coordinates to real-world millimetre coordinates via homography.

KEY FIX vs original
--------------------
* Camera is opened at a FIXED resolution (default 1280×720) that MUST
  match the resolution used in detect_snapshot.py.  Both values are
  stored inside calibration.npz so the detector can verify them at
  load time and warn if they differ.
* The window is shown in FIXED (not resizable) mode so mouse clicks
  always correspond to true image pixels.

Usage:
    python a4_calibration.py                     # webcam at 1280×720
    python a4_calibration.py --image photo.jpg   # static image
    python a4_calibration.py --camera 1          # camera index 1
    python a4_calibration.py --width 1920 --height 1080

Controls (corner-selection mode):
    Left-click   → Place a corner (order: BL → BR → TR → TL)
    Right-click  → Remove last placed corner
    ENTER        → Confirm corners & switch to measure mode
    R            → Reset corner selection
    ESC          → Quit

Controls (measure mode):
    Left-click   → Print real-world XY coords in mm
    S            → Save calibration to 'calibration.npz'
    L            → Load calibration from 'calibration.npz'
    R            → Redo corner selection
    ESC          → Quit
"""

import cv2
import numpy as np
import argparse
import os
import sys

# ── A4 dimensions in millimetres ──────────────────────────────────────────────
A4_W_MM = 210.0
A4_H_MM = 297.0

# ── BGR colours ───────────────────────────────────────────────────────────────
CLR_GREEN   = (0,   220,  80)
CLR_RED     = (0,    50, 220)
CLR_YELLOW  = (0,   210, 255)
CLR_WHITE   = (255, 255, 255)
CLR_CYAN    = (220, 200,   0)
CLR_OVERLAY = (20,   20,  20)


# ══════════════════════════════════════════════════════════════════════════════
class A4Calibrator:
    """Interactive A4-based homography calibrator."""

    CORNER_LABELS = [
        "BOTTOM-LEFT (0,0)",
        "BOTTOM-RIGHT (210,0)",
        "TOP-RIGHT (210,297)",
        "TOP-LEFT (0,297)",
    ]

    # World (mm) corners — origin at BOTTOM-LEFT, Y goes UP
    WORLD_PTS = np.array([
        [0,       0      ],   # BL
        [A4_W_MM, 0      ],   # BR
        [A4_W_MM, A4_H_MM],   # TR
        [0,       A4_H_MM],   # TL
    ], dtype=np.float32)

    def __init__(self, source, cap_width=1280, cap_height=720):
        self.source       = source
        self.cap_width    = cap_width
        self.cap_height   = cap_height
        self.pixel_pts    = []
        self.H            = None
        self.mode         = "corners"
        self.measure_pts  = []
        self.frame        = None
        self.orig_frame   = None
        self.cap          = None
        self.static_image = False

    # ── Source ────────────────────────────────────────────────────────────────
    def open_source(self):
        if isinstance(self.source, str):
            img = cv2.imread(self.source)
            if img is None:
                sys.exit(f"[ERROR] Cannot read image: {self.source}")
            self.orig_frame   = img.copy()
            self.cap_width    = img.shape[1]
            self.cap_height   = img.shape[0]
            self.static_image = True
            print(f"[INFO] Static image: {self.cap_width}×{self.cap_height} px")
        else:
            # ── CRITICAL: set resolution BEFORE reading any frame ────────────
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                sys.exit("[ERROR] Cannot open camera.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cap_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Confirm what the driver actually gave us
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w != self.cap_width or actual_h != self.cap_height:
                print(f"[WARN] Requested {self.cap_width}×{self.cap_height} but "
                      f"camera gave {actual_w}×{actual_h}.")
                print(f"[WARN] Updating calibration resolution to {actual_w}×{actual_h}.")
                self.cap_width  = actual_w
                self.cap_height = actual_h

            self.static_image = False
            print(f"[INFO] Camera opened at {self.cap_width}×{self.cap_height} px")

    def grab_frame(self):
        if self.static_image:
            return self.orig_frame.copy()
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.orig_frame = frame.copy()
        return frame

    # ── Homography ────────────────────────────────────────────────────────────
    def compute_homography(self):
        if len(self.pixel_pts) != 4:
            return False
        px = np.array(self.pixel_pts, dtype=np.float32)
        self.H, status = cv2.findHomography(px, self.WORLD_PTS)
        ok = self.H is not None and int(status.sum()) == 4
        if ok:
            print("\n[✓] Homography computed successfully.")
            print(f"    Mean reprojection error: {self._reproj_error():.4f} mm")
        return ok

    def _reproj_error(self):
        if self.H is None:
            return float("inf")
        px  = np.array(self.pixel_pts, dtype=np.float32).reshape(-1, 1, 2)
        mm  = cv2.perspectiveTransform(px, self.H)
        err = np.linalg.norm(mm.reshape(-1, 2) - self.WORLD_PTS, axis=1)
        return float(err.mean())

    def pixel_to_mm(self, px, py):
        if self.H is None:
            return None, None
        pt     = np.array([[[float(px), float(py)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        return float(result[0][0][0]), float(result[0][0][1])

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save_calibration(self, path="calibration.npz"):
        if self.H is None:
            print("[!] No calibration to save.")
            return
        np.savez(
            path,
            H          = self.H,
            pixel_pts  = np.array(self.pixel_pts, dtype=np.float32),
            # ── NEW: store the resolution so the detector can verify ──────────
            cap_width  = np.array(self.cap_width,  dtype=np.int32),
            cap_height = np.array(self.cap_height, dtype=np.int32),
        )
        print(f"[✓] Calibration saved → '{path}'")
        print(f"    Resolution stored: {self.cap_width}×{self.cap_height}")
        print(f"    Reprojection error: {self._reproj_error():.4f} mm")

    def load_calibration(self, path="calibration.npz"):
        if not os.path.exists(path):
            print(f"[!] File not found: {path}")
            return False
        data           = np.load(path)
        self.H         = data["H"]
        self.pixel_pts = data["pixel_pts"].tolist()
        self.mode      = "measure"
        if "cap_width" in data:
            stored_w = int(data["cap_width"])
            stored_h = int(data["cap_height"])
            print(f"[✓] Calibration loaded. Stored resolution: {stored_w}×{stored_h}")
            if stored_w != self.cap_width or stored_h != self.cap_height:
                print(f"[WARN] Current camera resolution {self.cap_width}×{self.cap_height} "
                      f"differs from stored {stored_w}×{stored_h}!")
                print("[WARN] XY coordinates will be WRONG unless resolutions match.")
        else:
            print("[✓] Calibration loaded (no resolution info stored — old format).")
        return True

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw_corner_ui(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), CLR_OVERLAY, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        next_idx = len(self.pixel_pts)
        msg = (f"Click: {self.CORNER_LABELS[next_idx]}  ({next_idx+1}/4)"
               if next_idx < 4 else "Press ENTER to confirm  |  R to reset")
        cv2.putText(frame, msg, (12, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, CLR_YELLOW, 1, cv2.LINE_AA)

        res_msg = f"Camera: {self.cap_width}x{self.cap_height}"
        cv2.putText(frame, res_msg, (w - 220, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        for i, (px, py) in enumerate(self.pixel_pts):
            cv2.circle(frame, (px, py), 10, CLR_GREEN, -1)
            cv2.circle(frame, (px, py), 12, CLR_WHITE, 1)
            cv2.putText(frame, self.CORNER_LABELS[i], (px + 14, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 1, cv2.LINE_AA)

        if len(self.pixel_pts) == 4:
            pts = np.array(self.pixel_pts, dtype=np.int32)
            cv2.polylines(frame, [pts], True, CLR_GREEN, 2, cv2.LINE_AA)

        cv2.putText(frame, "Right-click: undo last  |  ESC: quit",
                    (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    def _draw_measure_ui(self, frame):
        h, w = frame.shape[:2]

        if len(self.pixel_pts) == 4:
            pts = np.array(self.pixel_pts, dtype=np.int32)
            cv2.polylines(frame, [pts], True, CLR_GREEN, 2, cv2.LINE_AA)
            for px, py in self.pixel_pts:
                cv2.circle(frame, (px, py), 6, CLR_GREEN, -1)

        for (px, py, xmm, ymm) in self.measure_pts:
            cv2.circle(frame, (px, py), 7, CLR_RED, -1)
            cv2.circle(frame, (px, py), 9, CLR_WHITE, 1)
            cv2.putText(frame, f"({xmm:.1f}, {ymm:.1f}) mm", (px + 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_WHITE, 1, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), CLR_OVERLAY, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        err_str = (f"  |  reproj: {self._reproj_error():.3f} mm"
                   if self.H is not None else "")
        cv2.putText(frame,
                    f"[MEASURE]  Left-click → XY in mm{err_str}",
                    (12, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_CYAN, 1, cv2.LINE_AA)

        cv2.putText(frame,
                    "S: save  |  L: load  |  R: redo corners  |  ESC: quit",
                    (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Mouse ─────────────────────────────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if self.mode == "corners":
            if event == cv2.EVENT_LBUTTONDOWN and len(self.pixel_pts) < 4:
                self.pixel_pts.append([x, y])
                print(f"  Corner {len(self.pixel_pts)} ({self.CORNER_LABELS[len(self.pixel_pts)-1]}) "
                      f"→ pixel ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN and self.pixel_pts:
                removed = self.pixel_pts.pop()
                print(f"  Removed corner at pixel {removed}")
        elif self.mode == "measure":
            if event == cv2.EVENT_LBUTTONDOWN:
                xmm, ymm = self.pixel_to_mm(x, y)
                if xmm is not None:
                    self.measure_pts.append((x, y, xmm, ymm))
                    print(f"  Pixel ({x:4d}, {y:4d})  →  ({xmm:7.2f} mm, {ymm:7.2f} mm)")

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self.open_source()

        win = "A4 Calibration Tool"
        # FIXED window so mouse coords = pixel coords (no scaling artefacts)
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self._mouse_cb)

        print("\n══════════════════════════════════════════════════")
        print("  A4 Camera Calibration Tool")
        print("══════════════════════════════════════════════════")
        print(f"  Resolution : {self.cap_width} × {self.cap_height} px")
        print(f"  A4 size    : {A4_W_MM} × {A4_H_MM} mm")
        print("  Step 1 : Click the 4 A4 corners in this order:")
        print("           BOTTOM-LEFT (0,0) → BOTTOM-RIGHT → TOP-RIGHT → TOP-LEFT")
        print("  Step 2 : Press ENTER to compute homography")
        print("  Step 3 : Left-click anywhere to read mm coords")
        print("  Step 4 : Press S to save")
        print("══════════════════════════════════════════════════\n")

        while True:
            frame = self.grab_frame()
            if frame is None:
                break

            display = frame.copy()
            if self.mode == "corners":
                self._draw_corner_ui(display)
            else:
                self._draw_measure_ui(display)

            cv2.imshow(win, display)
            key = cv2.waitKey(1 if not self.static_image else 30) & 0xFF

            if key == 27:   # ESC
                break
            elif key in (13, 10):  # ENTER
                if self.mode == "corners":
                    if len(self.pixel_pts) == 4:
                        if self.compute_homography():
                            self.mode        = "measure"
                            self.measure_pts = []
                            print("\n[→] Measure mode. Left-click = mm coords.\n")
                        else:
                            print("[!] Homography failed — try clicking corners more carefully.")
                    else:
                        print(f"[!] Need 4 corners, have {len(self.pixel_pts)}.")
            elif key in (ord('r'), ord('R')):
                print("[↺] Resetting…")
                self.pixel_pts   = []
                self.H           = None
                self.mode        = "corners"
                self.measure_pts = []
            elif key in (ord('s'), ord('S')):
                self.save_calibration()
            elif key in (ord('l'), ord('L')):
                self.load_calibration()

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Headless helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_homography(path="calibration.npz"):
    data = np.load(path)
    return data["H"]

def pixel_to_mm(px, py, H):
    pt     = np.array([[[float(px), float(py)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0][0][0]), float(result[0][0][1])

def mm_to_pixel(x_mm, y_mm, H):
    H_inv  = np.linalg.inv(H)
    pt     = np.array([[[float(x_mm), float(y_mm)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H_inv)
    return int(result[0][0][0]), int(result[0][0][1])


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A4 Camera Calibration")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to a static image (omit to use webcam)")
    parser.add_argument("--camera", type=int, default=1,
                        help="Camera index (default: 1)")
    parser.add_argument("--width",  type=int, default=1280,
                        help="Capture width  (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Capture height (default: 720)")
    args = parser.parse_args()

    source = args.image if args.image else args.camera
    A4Calibrator(source, cap_width=args.width, cap_height=args.height).run()