"""
robot_controller_vision.py
────────────────────────────────────────────────────────────
Full pipeline: Vision (daylight HSV detector) + LLM planner + IK arm controller.

Key integration notes
─────────────────────
• Home position  = PHOTO_HOME_XYZ  (-50, -200, -50) in arm-space mm,
  passed through correct_arm_xy() before IK solve.
  The arm moves here before every camera capture so it is never in frame.
• camera_capture() moves the arm to the photo-home, grabs CAPTURE_FRAMES
  successive frames spaced CAPTURE_INTERVAL_S seconds apart, then returns
  all frames so the detector can fuse them.
• opencv_detect() accepts a list of frames, runs the HSV pipeline on each,
  fuses detections across frames using a ±MATCH_RADIUS_MM spatial tolerance,
  and resolves shape/color conflicts by majority vote.
• homography_transform() uses the loaded homography matrix (calibration.npz)
  via pixel_to_mm(), identical to detect_snapshot.py.
• XY axes are SWAPPED relative to raw homography output:
    arm_x  ← homography result[0][0][1]   (was row-axis)
    arm_y  ← homography result[0][0][0]   (was col-axis)
• Workspace filter: x ∈ (0, 280) mm  |  y ∈ (-210, 210) mm

Usage
─────
    python robot_controller_vision.py
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import ollama
import scipy.io.wavfile as wav
import serial
import serial.tools.list_ports
import sounddevice as sd
import whisper


# ══════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════

CAMERA_SRC        = 1              # camera index (int) or stream URL (str)
CAL_FILE          = "calibration.npz"
CAMERA_WARMUP     = 30             # frames to discard on camera open
INPUT_MODE        = "voice"         # "text" | "voice"
WHISPER_MODEL     = "base"
LLM_MODEL         = "gemma4:e4b"
SAMPLE_RATE       = 16_000
RECORD_SECONDS    = 5
CONFIDENCE_THRESH = 0.75           # min colour-fill ratio treated as "confidence"

# ── Multi-frame capture & fusion ─────────────────────────────────────────────
CAPTURE_FRAMES     = 5     # number of successive frames to grab per scene scan
CAPTURE_INTERVAL_S = 0.15  # seconds between successive frame grabs
# Two detections whose arm-space XY centres are within this radius (mm) across
# different frames are considered the same physical object.
MATCH_RADIUS_MM    = 5.0

# ── Photo-home position (arm clears the workspace for a clean camera shot) ──
# Coordinates are in arm-space mm BEFORE correct_arm_xy() correction.
PHOTO_HOME_XYZ = (-50.0, -220.0, -50.0)   # (x, y, z) in mm

# Zone definitions (mm, arm coordinate space, before IK correction)
ZONES: dict[str, dict] = {
    "left_zone":   {"arm_xy": (250.0, -150.0)},
    "right_zone":  {"arm_xy": (250.0,  150.0)},
    "center_zone": {"arm_xy": (250.0,   0.0)},
}

# ── Workspace bounds (mm, arm-space, after XY flip & homography) ─────────────
# Objects whose arm XY falls outside these limits are ignored.
WORKSPACE_X = (0.0,    280.0)   # x_min, x_max
WORKSPACE_Y = (-210.0, 210.0)   # y_min, y_max

SYSTEM_PROMPT = """
You control a robot arm on a flat surface viewed from above.

Zones on the surface:
  - left_zone:   (x=150 mm, y=-150 mm)
  - right_zone:  (x=150 mm, y= 150 mm)
  - center_zone: (x=150 mm, y=   0 mm)

Available primitives (use ONLY these):
  - pick(obj_id)       : pick up a detected object by its id
  - place(target_id)   : target_id can be a ZONE ID or an OBJECT ID

Rules:
  - When asked to move always pick before place
  - When asked to place always place and DO NOT pick beforehand
  - Never pick two objects simultaneously
  - Output ONLY a valid JSON array of steps, no explanation, no markdown
  - For targets use ONLY valid zones or obj_id
  - Make sure you pick the right color and shape
  - You may only pick or only place if the prompt instructs it

Output format:
[
  {"step": 1, "action": "pick",  "target": "target_id"},
  {"step": 2, "action": "place", "target": "target_id"},
  ...
]
"""


# ══════════════════════════════════════════════════════════════
#  HOMOGRAPHY  (pixel → mm)
#
#  AXIS CONVENTION (after the flip):
#    arm_x  ← homography col→row axis  (result[0][0][1])
#    arm_y  ← homography row→col axis  (result[0][0][0])
#
#  This matches the convention used throughout ArmController.
# ══════════════════════════════════════════════════════════════

H          = None
CAL_WIDTH  = 1280
CAL_HEIGHT = 720

if os.path.exists(CAL_FILE):
    cal_data = np.load(CAL_FILE)
    H = cal_data["H"].astype(np.float64)
    if "cap_width" in cal_data and "cap_height" in cal_data:
        CAL_WIDTH  = int(cal_data["cap_width"])
        CAL_HEIGHT = int(cal_data["cap_height"])
        print(f"[✓] Homography loaded from '{CAL_FILE}'  ({CAL_WIDTH}×{CAL_HEIGHT})")
    else:
        print(f"[✓] Homography loaded (old format — assuming {CAL_WIDTH}×{CAL_HEIGHT})")
        print("[WARN] Re-run a4_calibration.py if XY coords look wrong.")
else:
    print(f"[!] '{CAL_FILE}' not found — XY will be in pixels.")


def pixel_to_mm(px: int, py: int) -> Tuple[float, float]:
    """
    Convert camera pixel (px, py) → arm-space (arm_x_mm, arm_y_mm).

    Axes are deliberately swapped vs the raw homography output:
        arm_x  ← result[0][0][1]   (homography "row" dimension)
        arm_y  ← result[0][0][0]   (homography "col" dimension)
    """
    if H is None:
        return float(px), float(py)
    pt     = np.array([[[float(px), float(py)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    # ── FLIP: swap the two output axes ──────────────────────────────────────
    arm_x = round(float(result[0][0][1]), 2)   # was col, now treated as x
    arm_y = round(float(result[0][0][0]), 2)   # was row, now treated as y
    return arm_x, arm_y


# ══════════════════════════════════════════════════════════════
#  COLOUR DEFINITIONS — DAYLIGHT HSV RANGES
#  (matched to detect_snapshot.py for consistent detection)
# ══════════════════════════════════════════════════════════════

COLOR_RANGES = {
    "red": [
        (np.array([0,   130, 80]),  np.array([7,   255, 255])),
        (np.array([168, 130, 80]),  np.array([180, 255, 255])),
    ],
    "orange": [
        (np.array([8,   160, 80]),  np.array([20,  255, 255])),
        None,
    ],
    "yellow": [
        (np.array([21,  120, 80]),  np.array([34,  255, 255])),
        None,
    ],
    "green": [
        (np.array([35,   60, 50]),  np.array([85,  255, 255])),
        None,
    ],
    "blue": [
        (np.array([100,  80, 50]),  np.array([130, 255, 255])),
        None,
    ],
    "black": [
        (np.array([0,    0,   0]),  np.array([180, 255,  60])),
        None,
    ],
}

# Per-color fill thresholds — matched to detect_snapshot.py
COLOR_FILL_THRESHOLD = {
    "red":    0.20,
    "orange": 0.20,
    "yellow": 0.20,
    "green":  0.20,
    "blue":   0.20,
    "black":  0.10,
}

LABEL_COLORS = {
    "red":    (0,   0,   220),
    "orange": (0,   100, 255),
    "yellow": (0,   230, 255),
    "green":  (0,   200,  50),
    "blue":   (220, 100,   0),
    "black":  (180, 180, 180),
}


# ══════════════════════════════════════════════════════════════
#  VISION — DATA STRUCTURE
# ══════════════════════════════════════════════════════════════

@dataclass
class DetectedObject:
    shape:   str
    color:   str
    center:  Tuple[int, int]
    bbox:    Tuple[int, int, int, int]
    area:    float
    contour: np.ndarray = field(repr=False, default=None)


# ══════════════════════════════════════════════════════════════
#  VISION — COLOUR HELPERS
# ══════════════════════════════════════════════════════════════

def build_color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    ranges = COLOR_RANGES[color]
    mask   = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
    if ranges[1] is not None:
        mask |= cv2.inRange(hsv, ranges[1][0], ranges[1][1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def dominant_color_in_contour(hsv: np.ndarray, contour: np.ndarray) -> str:
    """
    Determine the dominant color inside a contour.
    Uses per-color fill thresholds from COLOR_FILL_THRESHOLD,
    matched to detect_snapshot.py (0.10 for black, 0.20 for all others).
    """
    h, w = hsv.shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)

    area = cv2.contourArea(contour)
    if area < 1:
        return "unknown"

    best_color, best_count = "unknown", 0
    for color in COLOR_RANGES:
        cmask    = build_color_mask(hsv, color)
        combined = cv2.bitwise_and(cmask, cmask, mask=region_mask)
        count    = cv2.countNonZero(combined)
        if count > best_count:
            best_count = count
            best_color = color

    threshold = COLOR_FILL_THRESHOLD.get(best_color, 0.20)
    if best_count < area * threshold:
        return "unknown"
    return best_color


# ══════════════════════════════════════════════════════════════
#  VISION — SHAPE CLASSIFIER
#  (matched to detect_snapshot.py — no solidity gate,
#   same circularity and vertex thresholds)
# ══════════════════════════════════════════════════════════════

MAX_RECT_ASPECT = 4.0


def classify_shape(contour: np.ndarray) -> Optional[str]:
    """
    Classify a contour as 'circle', 'square', or 'rectangle'.

    Logic mirrors detect_snapshot.py:
      - No solidity pre-filter (removed to match snapshot detector)
      - circularity > 0.85  → circle
      - 4 vertices           → square (aspect 0.85–1.15) or rectangle
      - vertices > 6 and circularity > 0.65 → circle
      - anything else        → None (rejected)
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return None

    area     = cv2.contourArea(contour)
    epsilon  = 0.03 * perimeter
    approx   = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    circ     = (4 * np.pi * area) / (perimeter ** 2)

    if circ > 0.85:
        return "circle"
    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h) if h > 0 else 0
        if aspect > MAX_RECT_ASPECT or aspect < 1.0 / MAX_RECT_ASPECT:
            return None
        return "square" if 0.85 <= aspect <= 1.15 else "rectangle"
    if vertices > 6 and circ > 0.65:
        return "circle"
    return None


# ══════════════════════════════════════════════════════════════
#  VISION — FRAME PROCESSOR
#  MIN_AREA and DEDUP_RADIUS matched to detect_snapshot.py
# ══════════════════════════════════════════════════════════════

MIN_AREA      = 600      # was 2000 — lowered to match detect_snapshot.py
MAX_AREA_FRAC = 0.50     # was 0.30 — raised to match detect_snapshot.py
DEDUP_RADIUS  = 30       # was 40  — lowered to match detect_snapshot.py


def process_frame(frame: np.ndarray) -> List[DetectedObject]:
    fh, fw   = frame.shape[:2]
    max_area = MAX_AREA_FRAC * fh * fw

    blurred = cv2.GaussianBlur(frame, (7, 7), 0)   # kernel matched to snapshot (7×7)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    detected: List[DetectedObject] = []
    seen_centers: List[Tuple[int, int]] = []

    for color in COLOR_RANGES:
        mask        = build_color_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > max_area:
                continue

            shape = classify_shape(cnt)
            if shape is None:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = x + bw // 2, y + bh // 2

            if any(abs(cx - sc[0]) < DEDUP_RADIUS and abs(cy - sc[1]) < DEDUP_RADIUS
                   for sc in seen_centers):
                continue

            confirmed_color = dominant_color_in_contour(hsv, cnt)
            if confirmed_color == "unknown":
                continue

            seen_centers.append((cx, cy))
            detected.append(DetectedObject(
                shape=shape, color=confirmed_color,
                center=(cx, cy), bbox=(x, y, bw, bh),
                area=area, contour=cnt,
            ))

    return detected


def draw_detections(frame: np.ndarray, objects: List[DetectedObject],
                    xy_unit: str = "mm") -> np.ndarray:
    out    = frame.copy()
    fh, fw = out.shape[:2]

    for obj in objects:
        bgr           = LABEL_COLORS.get(obj.color, (255, 255, 255))
        cx, cy        = obj.center
        x, y, bw, bh  = obj.bbox
        x_w, y_w      = pixel_to_mm(cx, cy)   # already flipped

        cv2.drawContours(out, [obj.contour], -1, bgr, 2)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), bgr, 1)
        cv2.drawMarker(out, (cx, cy), bgr,
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        label    = f"{obj.color} {obj.shape}"
        coord_w  = f"XY: ({x_w:.1f}, {y_w:.1f}) {xy_unit}"
        coord_px = f"px: ({cx}, {cy})"
        font, fs, th = cv2.FONT_HERSHEY_DUPLEX, 0.55, 1

        (lw, lh), _ = cv2.getTextSize(label,    font, fs,        th)
        (cw,  _), _ = cv2.getTextSize(coord_w,  font, fs - 0.10, 1)
        (pw,  _), _ = cv2.getTextSize(coord_px, font, fs - 0.15, 1)
        box_w = max(lw, cw, pw) + 10
        box_h = lh + 38

        lx = max(0, min(cx - box_w // 2, fw - box_w))
        ly = max(0, y - box_h - 4)

        overlay = out.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h),
                      (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

        cv2.putText(out, label,    (lx + 4, ly + lh + 4),
                    font, fs,        bgr,           th, cv2.LINE_AA)
        cv2.putText(out, coord_w,  (lx + 4, ly + lh + 18),
                    font, fs - 0.10, (180, 255, 180), 1,  cv2.LINE_AA)
        cv2.putText(out, coord_px, (lx + 4, ly + box_h - 5),
                    font, fs - 0.15, (160, 160, 160), 1,  cv2.LINE_AA)

    cv2.putText(out, f"Objects: {len(objects)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════
#  IK / ARM CONSTANTS
# ══════════════════════════════════════════════════════════════

L1 = 154.0
L2 = 150.0

EE_FORWARD = 160.0
EE_RIGHT   =  30.0

Z_APPROACH = 0.0
Z_GRIP     = -75.0
Z_GRIP2    = -50.0

SERVO_CAL: dict[str, tuple[float, float, int]] = {
    "theta1": ( 95.0,  0.0, +1),
    "theta2": (155.0, 90.0, +1),
    "theta3": (169.0,  0.0, +1),
    "theta4": ( 33.0,  0.0, +1),
}

SERVO_MIN = 0.0
SERVO_MAX = 200.0


# ══════════════════════════════════════════════════════════════
#  COORDINATE TRANSFORMS
# ══════════════════════════════════════════════════════════════

def correct_arm_xy(px: float, py: float) -> tuple[float, float]:
    x = 1.0913 * px - 8.7152 + 108
    if py <= -50:
        y = 1.0 * py - 12
    elif py >= 50:
        y = 1.22 * py - 5
    else:
        y = 1.1 * py + 0
    return x, y


# ══════════════════════════════════════════════════════════════
#  IK SOLVER
# ══════════════════════════════════════════════════════════════

def _math_limits() -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for joint, (s0, m0, d) in SERVO_CAL.items():
        a = m0 + (SERVO_MIN - s0) / d
        b = m0 + (SERVO_MAX - s0) / d
        limits[joint] = (min(a, b), max(a, b))
    return limits

MATH_LIMITS = _math_limits()


def math_to_servo(joint: str, math_angle: float) -> float:
    s0, m0, d = SERVO_CAL[joint]
    return max(SERVO_MIN, min(SERVO_MAX, s0 + d * (math_angle - m0)))


class IKSolver:
    def __init__(self, l1: float = L1, l2: float = L2):
        self.l1        = l1
        self.l2        = l2
        self.max_reach = l1 + l2 + 100
        self.min_reach = abs(l1 - l2)

    def _compute_math_angles(
            self, x: float, y: float, z: float, elbow_up: bool
    ) -> dict[str, float]:
        r = math.hypot(x, y)
        z += (200 - r) * 0.25

        d_xy_sq = x ** 2 + y ** 2
        d_xy    = math.sqrt(d_xy_sq)

        if d_xy_sq < EE_RIGHT ** 2:
            raise ValueError(f"Target ({x}, {y}) is inside the EE_RIGHT turning radius.")

        A  = math.sqrt(d_xy_sq - EE_RIGHT ** 2)
        rw = A - EE_FORWARD

        theta1_rad  = math.atan2(y, x) - math.atan2(EE_RIGHT, A)
        theta1_math = math.degrees(theta1_rad)

        dist_sq = rw ** 2 + z ** 2
        dist    = math.sqrt(dist_sq)

        if dist > self.max_reach:
            raise ValueError(f"Target unreachable: wrist {dist:.1f} mm > {self.max_reach}")
        if dist < self.min_reach:
            raise ValueError(f"Target too close: wrist {dist:.1f} mm < {self.min_reach}")

        cos3 = (dist_sq - self.l1 ** 2 - self.l2 ** 2) / (2.0 * self.l1 * self.l2)
        cos3 = max(-1.0, min(1.0, cos3))
        theta3_rad  = math.acos(cos3)
        if elbow_up:
            theta3_rad = -theta3_rad
        theta3_math = math.degrees(theta3_rad)

        theta2_rad  = math.atan2(z, rw) - math.atan2(
            self.l2 * math.sin(theta3_rad),
            self.l1 + self.l2 * math.cos(theta3_rad)
        )
        theta2_math = math.degrees(theta2_rad)
        theta4_math = -(theta2_math + theta3_math)-5

        return {
            "theta1": theta1_math,
            "theta2": theta2_math,
            "theta3": theta3_math,
            "theta4": theta4_math,
        }

    def _check_limits(self, math_angles: dict[str, float]) -> None:
        for joint, value in math_angles.items():
            lo, hi = MATH_LIMITS[joint]
            if not (lo <= value <= hi):
                needed_servo = math_to_servo(joint, value)
                raise ValueError(
                    f"{joint}: math angle {value:.1f}° → servo {needed_servo:.1f}° "
                    f"outside [{SERVO_MIN}°, {SERVO_MAX}°]."
                )

    def solve(
            self, x: float, y: float, z: float, elbow_up: bool = True
    ) -> tuple[dict[str, float], dict[str, float]]:
        math_angles  = self._compute_math_angles(x, y, z, elbow_up)
        self._check_limits(math_angles)
        servo_angles = {j: math_to_servo(j, v) for j, v in math_angles.items()}
        return math_angles, servo_angles


# ══════════════════════════════════════════════════════════════
#  ARM CONTROLLER
# ══════════════════════════════════════════════════════════════

class ArmController:
    BAUD_RATE  = 115_200
    MOVE_DELAY = 1.2

    def __init__(self, port: str | None = None, baud: int = BAUD_RATE):
        self.solver      = IKSolver()
        self.ser: serial.Serial | None = None
        self._gripper    = 0
        self._current_xy: tuple[float, float] | None = None

        if port is None:
            port = self._auto_detect_port()
        self._connect(port, baud)

    # ── connection ────────────────────────────────────────────

    @staticmethod
    def _auto_detect_port() -> str:
        candidates = []
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if any(kw in desc for kw in ("arduino", "ch340", "cp210", "ftdi", "acm", "usb serial")):
                candidates.append(p.device)
        if not candidates:
            all_ports = [p.device for p in serial.tools.list_ports.comports()]
            raise RuntimeError(
                f"No Arduino port found. Available: {all_ports}\n"
                "Pass port explicitly: ArmController(port='COM3')"
            )
        print(f"[serial] Auto-detected port: {candidates[0]}")
        return candidates[0]

    def _connect(self, port: str, baud: int) -> None:
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print(f"[serial] Connected to {port} @ {baud} baud")

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[serial] Disconnected")

    # ── low-level send ────────────────────────────────────────

    def _send(self, servo_angles: dict[str, float], gripper: int | None = None) -> None:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("Serial port is not open.")
        if gripper is not None:
            self._gripper = int(bool(gripper))

        packet = (
            f"{servo_angles['theta1']:.1f},"
            f"{servo_angles['theta2']:.1f},"
            f"{servo_angles['theta3']:.1f},"
            f"{servo_angles['theta4']:.1f},"
            f"{self._gripper}\n"
        )
        self.ser.write(packet.encode("ascii"))
        print(
            f"  [pkt]  T1={servo_angles['theta1']:5.1f}°  "
            f"T2={servo_angles['theta2']:5.1f}°  "
            f"T3={servo_angles['theta3']:5.1f}°  "
            f"T4={servo_angles['theta4']:5.1f}°  "
            f"grip={'CLOSE' if self._gripper else 'open'}"
        )

    # ── movement helpers ──────────────────────────────────────

    def _move(
        self,
        x: float, y: float, z: float,
        gripper: int | None = None,
        elbow_up: bool = True,
        label: str = "",
    ) -> None:
        _, servo_a = self.solver.solve(x, y, z, elbow_up=elbow_up)
        if label:
            print(f"  [{label}]  ({x:.0f}, {y:.0f}, {z:.0f}) mm")
        self._send(servo_a, gripper=gripper)
        time.sleep(self.MOVE_DELAY)
        self._current_xy = (x, y)

    def _transit(
            self,
            x_to: float, y_to: float,
            gripper: int | None = None,
            elbow_up: bool = True,
    ) -> None:
        if self._current_xy is None:
            return
        x_from, y_from = self._current_xy
        dx = x_to - x_from
        dy = y_to - y_from
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist < 1.0:
            return

        n_segments = max(1, int(math.ceil(dist / 3.0)))

        print(f"  [transit] {x_from:.0f},{y_from:.0f} → {x_to:.0f},{y_to:.0f}  "
              f"dist={dist:.0f} mm  waypoints={n_segments}")

        for i in range(1, n_segments + 1):
            t = i / n_segments
            wx = x_from + dx * t
            wy = y_from + dy * t
            try:
                math_a, servo_a = self.solver.solve(wx, wy, Z_APPROACH, elbow_up=elbow_up)
                if gripper is not None:
                    self._gripper = int(bool(gripper))
                packet = (
                    f"{servo_a['theta1']:.1f},"
                    f"{servo_a['theta2']:.1f},"
                    f"{servo_a['theta3']:.1f},"
                    f"{servo_a['theta4']:.1f},"
                    f"{self._gripper}\n"
                )
                self.ser.write(packet.encode("ascii"))
                time.sleep(0.015)
                self._current_xy = (wx, wy)
            except ValueError:
                pass

    def _set_gripper(self, state: int, servo_angles: dict[str, float]) -> None:
        print(f"  [{'CLOSING' if state else 'OPENING'} gripper]")
        self._send(servo_angles, gripper=state)
        time.sleep(0.6)

    # ── photo home ────────────────────────────────────────────

    def move_to_photo_home(self) -> None:
        """
        Move arm to PHOTO_HOME_XYZ (-50, -220, -50).
        x and y are passed through correct_arm_xy() so the polynomial
        correction is consistently applied.
        """
        raw_x, raw_y, z = PHOTO_HOME_XYZ
        x, y = correct_arm_xy(raw_x, raw_y)
        print(f"[arm] Moving to photo-home  raw=({raw_x}, {raw_y}, {z})  "
              f"corrected=({x:.1f}, {y:.1f}, {z}) mm …")

        if self._current_xy is not None:
            self._transit(x, y, gripper=self._gripper, elbow_up=True)

        try:
            _, servo_a = self.solver.solve(x, y, z, elbow_up=True)
            self._send(servo_a, gripper=self._gripper)
            time.sleep(self.MOVE_DELAY)
            self._current_xy = (x, y)
            print("[arm] Photo-home reached ✓")
        except ValueError as e:
            print(f"[arm] ✗ Cannot reach photo-home: {e}")
            print("[arm] Sending raw HOME servo positions instead.")
            home_servo = {j: SERVO_CAL[j][0] for j in SERVO_CAL}
            self._send(home_servo, gripper=0)
            time.sleep(self.MOVE_DELAY)

    # ── pick / place ──────────────────────────────────────────

    def pick(self, x: float, y: float, elbow_up: bool = True) -> None:
        x, y = correct_arm_xy(x, y)
        print(f"[pick]  corrected x={x:.1f}  y={y:.1f} mm")
        self._transit(x, y, gripper=0, elbow_up=elbow_up)
        self._move(x, y, Z_APPROACH, gripper=0, elbow_up=elbow_up, label="approach")
        self._move(x, y, Z_GRIP,     gripper=0, elbow_up=elbow_up, label="descend ")
        _, servo_grip = self.solver.solve(x, y, Z_GRIP, elbow_up=elbow_up)
        self._set_gripper(1, servo_grip)
        self._move(x, y, Z_APPROACH, gripper=1, elbow_up=elbow_up, label="retract ")
        print("[pick]  done ✓")

    def place(self, x: float, y: float, elbow_up: bool = True) -> None:
        x, y = correct_arm_xy(x, y)
        print(f"[place] corrected x={x:.1f}  y={y:.1f} mm")
        self._transit(x, y, gripper=1, elbow_up=elbow_up)
        self._move(x, y, Z_APPROACH, gripper=1, elbow_up=elbow_up, label="approach")
        self._move(x, y, Z_GRIP2,    gripper=1, elbow_up=elbow_up, label="descend ")
        _, servo_rel = self.solver.solve(x, y, Z_GRIP2, elbow_up=elbow_up)
        self._set_gripper(0, servo_rel)
        self._move(x, y, Z_APPROACH, gripper=0, elbow_up=elbow_up, label="retract ")
        print("[place] done ✓")

    # ── context manager ───────────────────────────────────────

    def __enter__(self):  return self
    def __exit__(self, *_): self.disconnect()


# ══════════════════════════════════════════════════════════════
#  CAMERA — opens once, reused across captures
# ══════════════════════════════════════════════════════════════

_cap: cv2.VideoCapture | None = None


def open_camera() -> cv2.VideoCapture:
    global _cap
    if isinstance(CAMERA_SRC, int):
        _cap = cv2.VideoCapture(CAMERA_SRC, cv2.CAP_DSHOW)
    else:
        _cap = cv2.VideoCapture(CAMERA_SRC)

    if not _cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {CAMERA_SRC}")

    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAL_WIDTH)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAL_HEIGHT)
    _cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] Opened at {actual_w}×{actual_h} px  (cal: {CAL_WIDTH}×{CAL_HEIGHT})")
    if actual_w != CAL_WIDTH or actual_h != CAL_HEIGHT:
        print("[WARN] Resolution mismatch — XY coords may be wrong. Re-run a4_calibration.py.")

    print(f"[camera] Warming up ({CAMERA_WARMUP} frames)…")
    for _ in range(CAMERA_WARMUP):
        _cap.read()
    print("[camera] Ready.")
    return _cap


def _grab_raw_frame() -> np.ndarray:
    """Flush the camera ring-buffer and return one fresh BGR frame."""
    global _cap
    if _cap is None or not _cap.isOpened():
        open_camera()
    for _ in range(5):
        _cap.read()
    ret, frame = _cap.read()
    if not ret or frame is None:
        raise RuntimeError("[camera] Failed to grab frame.")
    return frame


def camera_capture(arm: ArmController) -> list[np.ndarray]:
    """
    1. Move arm to photo-home so it is out of frame.
    2. Let vibration settle, then grab CAPTURE_FRAMES successive frames
       spaced CAPTURE_INTERVAL_S seconds apart.
    3. Return the list of BGR frames — arm stays at photo-home.
    """
    arm.move_to_photo_home()
    time.sleep(0.4)

    frames: list[np.ndarray] = []
    for i in range(CAPTURE_FRAMES):
        frame = _grab_raw_frame()
        frames.append(frame)
        print(f"[camera] Frame {i + 1}/{CAPTURE_FRAMES} captured  "
              f"({frame.shape[1]}×{frame.shape[0]})")
        if i < CAPTURE_FRAMES - 1:
            time.sleep(CAPTURE_INTERVAL_S)

    return frames


# ══════════════════════════════════════════════════════════════
#  OPENCV DETECT
# ══════════════════════════════════════════════════════════════

def _detect_single_frame(frame: np.ndarray) -> list[dict]:
    """
    Run the HSV shape/colour pipeline on one frame and return raw result dicts
    with arm_xy already populated. Workspace-out-of-bounds objects are dropped.
    """
    objects_raw = process_frame(frame)
    results: list[dict] = []

    for obj in objects_raw:
        cx, cy   = obj.center
        x_w, y_w = pixel_to_mm(cx, cy)

        x_ok = WORKSPACE_X[0] <= x_w <= WORKSPACE_X[1]
        y_ok = WORKSPACE_Y[0] <= y_w <= WORKSPACE_Y[1]
        if not x_ok or not y_ok:
            continue

        results.append({
            "shape":      obj.shape,
            "color":      obj.color,
            "pixel":      [cx, cy],
            "arm_xy":     [x_w, y_w],
            "area_pixel": int(obj.area),
            "contour":    obj.contour,
        })

    return results


def fuse_multi_frame_detections(
    per_frame: list[list[dict]],
) -> list[dict]:
    """
    Merge detections from multiple frames into a stable object list.

    Algorithm
    ---------
    For every detection in every frame we attempt to assign it to an existing
    cluster whose centroid is within MATCH_RADIUS_MM mm.  If none matches we
    open a new cluster.  After all frames are consumed each cluster resolves
    its final shape and color by majority vote (ties broken by first seen).
    The returned arm_xy is the mean centroid across all frames that contributed
    to the cluster.
    """
    clusters: list[dict] = []

    for frame_idx, detections in enumerate(per_frame):
        for det in detections:
            dx, dy = det["arm_xy"]

            best_idx, best_dist = -1, float("inf")
            for ci, cl in enumerate(clusters):
                cx = cl["arm_xy_sum"][0] / cl["count"]
                cy = cl["arm_xy_sum"][1] / cl["count"]
                dist = ((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = ci

            if best_idx >= 0 and best_dist <= MATCH_RADIUS_MM:
                cl = clusters[best_idx]
                cl["arm_xy_sum"][0] += dx
                cl["arm_xy_sum"][1] += dy
                cl["count"]         += 1
                cl["shapes"].append(det["shape"])
                cl["colors"].append(det["color"])
                if det["area_pixel"] > cl["best_area"]:
                    cl["best_area"]    = det["area_pixel"]
                    cl["best_contour"] = det.get("contour")
                    cl["best_pixel"]   = det["pixel"]
            else:
                clusters.append({
                    "arm_xy_sum":   [dx, dy],
                    "count":        1,
                    "shapes":       [det["shape"]],
                    "colors":       [det["color"]],
                    "best_area":    det["area_pixel"],
                    "best_contour": det.get("contour"),
                    "best_pixel":   det["pixel"],
                })

    fused: list[dict] = []
    for ci, cl in enumerate(clusters, start=1):
        cx = round(cl["arm_xy_sum"][0] / cl["count"], 2)
        cy = round(cl["arm_xy_sum"][1] / cl["count"], 2)

        def majority(seq: list[str]) -> str:
            counts: dict[str, int] = {}
            for v in seq:
                counts[v] = counts.get(v, 0) + 1
            return max(counts, key=lambda k: (counts[k], -seq.index(k)))

        shape = majority(cl["shapes"])
        color = majority(cl["colors"])
        votes_shape = {s: cl["shapes"].count(s) for s in set(cl["shapes"])}
        votes_color = {c: cl["colors"].count(c) for c in set(cl["colors"])}

        print(
            f"  [fusion] cluster {ci}: {color} {shape}  "
            f"XY=({cx:.1f}, {cy:.1f}) mm  "
            f"seen={cl['count']} frames  "
            f"shape_votes={votes_shape}  color_votes={votes_color}"
        )

        fused.append({
            "id":          f"obj_{ci}",
            "name":        f"{color}_{shape}",
            "shape":       shape,
            "color":       color,
            "pixel":       cl["best_pixel"],
            "confidence":  round(COLOR_FILL_THRESHOLD.get(color, 0.20) + 0.70, 2),
            "arm_xy":      [cx, cy],
            "area_pixel":  cl["best_area"],
            "frames_seen": cl["count"],
            "_contour":    cl["best_contour"],
        })

    return fused


def opencv_detect(frames: list[np.ndarray]) -> list[dict]:
    """
    Run the daylight shape/colour pipeline over multiple frames, fuse
    results across frames, and return a stable list of object dicts.

    Workspace filter (post XY-flip): x ∈ (0, 280) mm | y ∈ (-210, 210) mm

    Side-effects:
      * Displays the annotated LAST frame in a window (press any key to close).
      * Saves snapshot_debug.jpg to disk.
      * Prints the fused detections JSON to stdout.
    """
    xy_unit = "mm" if H is not None else "px"

    per_frame: list[list[dict]] = []
    for i, frame in enumerate(frames):
        dets = _detect_single_frame(frame)
        per_frame.append(dets)
        print(f"[vision] Frame {i + 1}: {len(dets)} object(s) in workspace")

    print(f"[vision] Fusing {len(frames)} frames  (match radius ±{MATCH_RADIUS_MM} mm) …")
    fused = fuse_multi_frame_detections(per_frame)
    print(f"[vision] {len(fused)} unique object(s) after fusion")

    last_frame = frames[-1]
    annotated  = last_frame.copy()
    fh, fw     = annotated.shape[:2]

    for obj in fused:
        if obj["_contour"] is None:
            continue
        bgr         = LABEL_COLORS.get(obj["color"], (255, 255, 255))
        cx, cy      = obj["pixel"]
        x_w, y_w    = obj["arm_xy"]
        frames_seen = obj["frames_seen"]
        contour     = obj["_contour"]

        cv2.drawContours(annotated, [contour], -1, bgr, 2)
        cv2.drawMarker(annotated, (cx, cy), bgr,
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        label   = f"{obj['color']} {obj['shape']}  ({frames_seen}/{len(frames)}f)"
        coord_w = f"XY: ({x_w:.1f}, {y_w:.1f}) {xy_unit}"
        font, fs, th = cv2.FONT_HERSHEY_DUPLEX, 0.55, 1

        (lw, lh), _ = cv2.getTextSize(label,   font, fs,        th)
        (cw,  _), _ = cv2.getTextSize(coord_w, font, fs - 0.10, 1)
        box_w = max(lw, cw) + 10
        box_h = lh + 26
        lx = max(0, min(cx - box_w // 2, fw - box_w))
        ly = max(0, cy - box_h - 4)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h),
                      (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
        cv2.putText(annotated, label,   (lx + 4, ly + lh + 4),
                    font, fs,        bgr,           th, cv2.LINE_AA)
        cv2.putText(annotated, coord_w, (lx + 4, ly + lh + 18),
                    font, fs - 0.10, (180, 255, 180), 1, cv2.LINE_AA)

    cv2.putText(annotated,
                f"Objects: {len(fused)}  (fused over {len(frames)} frames)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.imwrite("snapshot_debug.jpg", annotated)
    print("[vision] Snapshot saved → snapshot_debug.jpg")

    cv2.imshow("Detection result -- press any key to continue", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    results = [{k: v for k, v in obj.items() if k != "_contour"} for obj in fused]

    json_str = json.dumps(results, indent=2)
    print("\n[vision] Fused detections JSON:")
    print(json_str)
    print()

    return results


# ══════════════════════════════════════════════════════════════
#  STT
# ══════════════════════════════════════════════════════════════

def listen_for_command(whisper_model) -> str:
    """
    Press Enter to start recording, then Enter again to stop.
    Audio is captured on a background thread so the main thread
    can watch for the second Enter without blocking sd.rec().
    """
    import threading

    audio_path = os.path.join(os.getcwd(), "command.wav")

    input("\n[STT] Press Enter to start recording...")
    print("[STT] Recording... Press Enter to stop.")

    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def _record():
        chunk_frames = int(SAMPLE_RATE * 0.1)
        while not stop_event.is_set():
            chunk = sd.rec(chunk_frames, samplerate=SAMPLE_RATE,
                           channels=1, dtype="int16", blocking=True)
            chunks.append(chunk)

    recorder = threading.Thread(target=_record, daemon=True)
    recorder.start()

    input()
    stop_event.set()
    recorder.join(timeout=1.0)

    if not chunks:
        print("[STT] No audio captured.")
        return ""

    try:
        audio = np.concatenate(chunks, axis=0)
        wav.write(audio_path, SAMPLE_RATE, audio)
        print(f"[STT] Recorded {len(audio) / SAMPLE_RATE:.1f} s — transcribing...")
        result  = whisper_model.transcribe(audio_path, fp16=False)
        command = result["text"].strip()
        print(f"[STT] Heard: '{command}'")
        return command
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""


def get_user_command(whisper_model=None) -> str:
    if INPUT_MODE.lower() == "voice":
        if whisper_model is None:
            raise ValueError("Whisper model required for voice mode.")
        return listen_for_command(whisper_model)
    return input("\n[TEXT] Enter command for the robot: ").strip()


# ══════════════════════════════════════════════════════════════
#  LLM PLANNER
# ══════════════════════════════════════════════════════════════

def build_scene_description(objects: list[dict]) -> str:
    lines = [
        f"  {o['id']}: {o['color']} {o['shape']} "
        f"at arm_xy=({o['arm_xy'][0]:.1f}, {o['arm_xy'][1]:.1f}) mm"
        for o in objects
    ]
    return "\n".join(lines)


def call_llm_planner(scene_description: str, command: str) -> list[dict]:
    print("[LLM] Generating plan…")
    user_message = f"Scene:\n{scene_description}\n\nCommand: \"{command}\""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        options={"num_ctx": 4096, "num_gpu": 42, "temperature": 0},
    )

    eval_count    = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    if eval_count and eval_duration:
        tps = eval_count / (eval_duration / 1e9)
        print(f"[LLM] {tps:.2f} tok/s  |  {eval_count} tokens")
        if tps < 15:
            print("⚠️  [LLM] Low speed — model may be spilling to CPU.")

    raw = response["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    print(f"[LLM] Raw response:\n{raw}")

    plan = json.loads(raw)
    print(f"[LLM] Plan has {len(plan)} step(s)")
    return plan


# ══════════════════════════════════════════════════════════════
#  EXECUTOR
# ══════════════════════════════════════════════════════════════

def execute_plan(
    plan: list[dict],
    detected_objects: list[dict],
    arm: ArmController,
) -> str:
    obj_map = {o["id"]: o for o in detected_objects}

    for step in plan:
        action    = step["action"]
        target_id = step.get("target")
        print(f"\n[Executor] Step {step['step']}: {action} → {target_id}")

        if action == "pick":
            obj = obj_map.get(target_id)
            if obj is None:
                print(f"[Executor] ✗ Object '{target_id}' not found in scene.")
                return "abort"
            x, y = obj["arm_xy"]
            try:
                arm.pick(x, y)
            except ValueError as e:
                print(f"[Executor] ✗ IK failed for pick: {e}")
                return "abort"

        elif action == "place":
            dest = ZONES.get(target_id) or obj_map.get(target_id)
            if dest is None:
                print(f"[Executor] ✗ '{target_id}' is not a valid zone or object.")
                return "abort"
            x, y = dest["arm_xy"]
            try:
                arm.place(x, y)
            except ValueError as e:
                print(f"[Executor] ✗ IK failed for place: {e}")
                return "abort"

        else:
            print(f"[Executor] ✗ Unknown action '{action}' — skipping.")

    print("[Executor] Sequence complete ✓")
    return "done"


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    print(f"=== Robot Controller + Vision  (mode: {INPUT_MODE.upper()}) ===")

    whisper_mdl = None
    if INPUT_MODE.lower() == "voice":
        print("Loading Whisper…")
        whisper_mdl = whisper.load_model(WHISPER_MODEL)

    open_camera()

    print("Connecting to arm…")
    with ArmController() as arm:
        print(f"\n[init] Moving to photo-home {PHOTO_HOME_XYZ} …")
        arm.move_to_photo_home()
        print("System ready.\n")

        while True:
            command = get_user_command(whisper_mdl)
            if not command or len(command) < 3:
                continue
            if command.lower() in ("exit", "quit"):
                break

            print(f"\n[Main] Capturing scene ({CAPTURE_FRAMES} frames)…")
            try:
                frames = camera_capture(arm)
            except RuntimeError as e:
                print(f"[Main] Camera error: {e}")
                continue

            print("[Main] Running detection…")
            detected_objects = opencv_detect(frames)

            if not detected_objects:
                print("[Main] No objects detected — skipping plan.")
                continue

            scene_desc = build_scene_description(detected_objects)
            print(f"\n[Main] Scene:\n{scene_desc}")

            try:
                plan = call_llm_planner(scene_desc, command)
            except Exception as e:
                print(f"[Main] LLM error: {e}")
                continue

            result = execute_plan(plan, detected_objects, arm)
            print(f"\n[Main] Execution result: {result}")

            arm.move_to_photo_home()

    print("Goodbye.")


if __name__ == "__main__":
    main()