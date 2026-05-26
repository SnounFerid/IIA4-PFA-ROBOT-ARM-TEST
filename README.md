# VoiceArm — LLM-Driven Robotic Arm with Computer Vision

**VoiceArm** is an end-to-end pipeline that accepts spoken or typed natural-language commands, interprets them using a locally-running large language model, detects physical objects on a workspace surface using a top-mounted camera, and executes pick-and-place operations on a 6-DOF servo arm — all without cloud services or an internet connection.

---

## Table of Contents

- [Team](#team)
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Hardware](#hardware)
- [Software Components](#software-components)
  - [IK1.ino — Arduino Firmware](#ik1ino--arduino-firmware)
  - [a4_calibration.py — Camera Calibration Tool](#a4_calibrationpy--camera-calibration-tool)
  - [finalv1.py — Main Controller Pipeline](#finalv1py--main-controller-pipeline)
- [Installation](#installation)
- [Setup and Usage](#setup-and-usage)
- [Configuration Reference](#configuration-reference)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Team

| Name | Role |
|------|------|
| Snoun Ferid | Project Lead |
| Souheib Khelil | Team Member |
| Elyes Grati | Team Member |
| Oussema Njeh | Team Member |

---

## Project Overview

VoiceArm connects five layers of technology into a single working system:

1. **Speech recognition** — OpenAI Whisper captures and transcribes user voice commands locally.
2. **Scene perception** — An overhead camera detects objects on the workspace, classifying each one by color (red, orange, yellow, green, blue, black) and shape (circle, square, rectangle) using HSV-based computer vision.
3. **LLM planning** — A local language model (Gemma 4 via Ollama) receives a text description of the detected scene plus the user's command and outputs a JSON action plan using only two primitives: `pick` and `place`.
4. **Inverse kinematics** — A Python IK solver converts XYZ millimetre coordinates into four joint angles for a 2-link planar arm with end-effector offset, then applies per-joint servo calibration offsets.
5. **Firmware execution** — An Arduino reads serial packets of servo angles and drives six servos with a step-rate limiter that prevents jerky motion.

The 3D-printed arm model is sourced from the open-source repository [doganhobby/Arduino-Projects](https://github.com/doganhobby/Arduino-Projects/tree/main/3D%20Robot%20Arm). All control software — firmware, calibration tool, vision pipeline, IK solver, LLM interface, and execution loop — was written entirely by the VoiceArm team.

---

## System Architecture

```
User voice / text
       |
       v
  OpenAI Whisper (local STT)
       |
       v
  Camera capture (arm moves to photo-home first)
       |
       v
  OpenCV vision pipeline
  (HSV color detection, shape classification, homography transform)
       |
       v
  Ollama LLM planner (Gemma 4)
  (receives scene description + command, outputs JSON plan)
       |
       v
  Python IK solver
  (XYZ mm -> joint angles -> servo calibration -> serial packet)
       |
       v
  Arduino (IK1.ino)
  (step-rate-limited servo control over USB serial)
       |
       v
  6-DOF servo arm
```

---

## Hardware

| Component | Details |
|-----------|---------|
| Microcontroller | Arduino (Uno or compatible) |
| Servos | 6x hobby servos (base, shoulder x2, elbow, wrist, gripper) |
| Camera | USB webcam, mounted overhead looking down at the workspace |
| Computer | Any machine capable of running Python 3.11+ and a local LLM (GPU recommended) |
| Reference sheet | A4 paper (210 x 297 mm) used for camera calibration |
| 3D-printed arm | Model from [doganhobby/Arduino-Projects](https://github.com/doganhobby/Arduino-Projects/tree/main/3D%20Robot%20Arm) |

Servo pin assignments on the Arduino:

| Servo | Pin |
|-------|-----|
| Shoulder 1 | 3 |
| Shoulder 2 | 5 |
| Base | 6 |
| Elbow | 9 |
| Wrist | 10 |
| Gripper | 11 |

---

## Software Components

### IK1.ino — Arduino Firmware

**Purpose:** Receives comma-separated servo angle packets over USB serial and drives six servos with smooth, rate-limited motion.

**How it works:**

The firmware maintains two arrays per servo joint: a *target* position and a *current* (smoothed) position. On every loop tick (every 10 ms by default), each servo is advanced by at most 1 degree toward its target. This rate limiter (`STEP_SIZE = 1`, `INTERVAL = 10 ms`) caps movement at approximately 100 degrees per second, preventing abrupt mechanical stress and protecting the servos from overloading.

Incoming serial packets arrive as a newline-terminated string with five comma-separated integers:

```
<base>,<shoulder>,<elbow>,<wrist>,<gripper>\n
```

The shoulder joint uses a mirrored second servo (`shoulder2 = 180 - shoulder1`) so that both servos work together mechanically without requiring independent angle values from the host. The gripper is driven by boolean logic: a value of `1` commands the gripper to close (servo position 85) and `0` commands it to open (servo position 145).

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STEP_SIZE` | 1 | Maximum degrees moved per firmware tick |
| `INTERVAL` | 10 ms | Time between ticks |
| Serial baud | 115200 | Must match Python host |

---

### a4_calibration.py — Camera Calibration Tool

**Purpose:** Establishes the mathematical mapping (homography) between camera pixel coordinates and real-world millimetre coordinates in arm space, using a standard A4 sheet as a reference target.

**How it works:**

The user places an A4 sheet (210 x 297 mm) flat in the camera field of view. The tool opens an interactive OpenCV window and asks the user to click the four corners of the sheet in a fixed order: bottom-left, bottom-right, top-right, top-left. Each click records a pixel coordinate paired with a known real-world millimetre coordinate.

Once all four corners are placed, the tool calls `cv2.findHomography()` to compute a 3x3 perspective transformation matrix `H` that maps any pixel `(px, py)` to its corresponding position in millimetres. The reprojection error (in millimetres) is reported to the user so calibration quality can be assessed.

The computed matrix and the camera resolution at which it was produced are saved to `calibration.npz`. At load time, the detector checks that the active camera resolution matches the stored resolution and warns the user if they differ, because resolution mismatches cause incorrect coordinate mapping.

**Controls:**

| Key / Action | Effect |
|--------------|--------|
| Left-click (corner mode) | Place next corner |
| Right-click (corner mode) | Undo last corner |
| Enter | Confirm corners, compute homography |
| S | Save calibration to `calibration.npz` |
| L | Load calibration from `calibration.npz` |
| R | Reset to corner selection |
| Esc | Quit |

**Command-line usage:**

```bash
# Use webcam at default 1280x720
python a4_calibration.py

# Use a static image
python a4_calibration.py --image photo.jpg

# Specify camera index
python a4_calibration.py --camera 1

# Specify resolution
python a4_calibration.py --width 1920 --height 1080
```

The module also exposes three headless helper functions for use by other scripts:

- `load_homography(path)` — loads `H` from a `.npz` file
- `pixel_to_mm(px, py, H)` — converts pixel to millimetres
- `mm_to_pixel(x_mm, y_mm, H)` — converts millimetres back to pixels (via `H` inverse)

---

### finalv1.py — Main Controller Pipeline

This is the entry point for the complete system. It integrates five subsystems: camera management, computer vision, speech recognition, LLM planning, and arm control.

#### 1. Camera Management

The camera is opened once at startup and kept alive throughout the session. A configurable warm-up period (`CAMERA_WARMUP = 30` frames) discards the initial unstable frames that webcams produce on first open.

Before every capture, `move_to_photo_home()` drives the arm to a fixed out-of-frame position (`PHOTO_HOME_XYZ = (-50, -220, -50)` mm) so the arm does not occlude the workspace in the image. The system then captures `CAPTURE_FRAMES = 5` successive frames, spaced `CAPTURE_INTERVAL_S = 0.15` seconds apart, for multi-frame fusion.

#### 2. Computer Vision Pipeline

The vision system operates in HSV colour space rather than RGB because HSV is significantly more robust to changes in lighting conditions.

**Colour detection:** For each supported colour, the pipeline builds an HSV range mask. Red wraps around the HSV hue circle and therefore uses two ranges combined with a bitwise OR. All masks are cleaned with morphological opening (removes noise) followed by closing (fills small gaps).

**Shape classification (`classify_shape`):** For each detected contour the function computes circularity `= 4 * pi * area / perimeter^2`. A circularity above 0.85 is classified as a circle. A four-vertex polygon approximation with an aspect ratio between 0.85 and 1.15 is classified as a square; outside that range it becomes a rectangle. High-vertex polygons with circularity above 0.65 are also classified as circles (e.g. camera-blurred circular objects with irregular outlines).

**Color confirmation (`dominant_color_in_contour`):** After shape detection, the color assignment is double-checked by painting only the interior of the contour as a mask, then computing how many HSV-matching pixels of each color appear inside it. Only the dominant color exceeding its fill threshold (20% of contour area for most colors, 10% for black) is accepted.

**Multi-frame fusion:** Detections from all captured frames are merged. Two detections from different frames whose arm-space XY centres are within `MATCH_RADIUS_MM = 5.0` mm of each other are considered the same physical object. Shape and color labels are resolved by majority vote across frames, making the system robust against single-frame misdetections.

**Coordinate transform:** Each detected object's pixel centre is passed through `pixel_to_mm()`, which applies the homography matrix and then deliberately swaps the two output axes (the raw homography column axis becomes arm X, and the row axis becomes arm Y) to match the physical orientation of the workspace relative to the camera.

#### 3. Speech Recognition

If `INPUT_MODE = "voice"`, the system uses OpenAI Whisper running locally. The user presses Enter to begin recording, speaks the command, and presses Enter again to stop. Audio is captured on a background thread using `sounddevice` so the main thread can remain responsive to the stop signal. The recorded audio is saved as a WAV file and transcribed by Whisper.

If `INPUT_MODE = "text"`, the system simply reads from standard input.

#### 4. LLM Planner

The planner sends two messages to the local LLM via `ollama.chat()`:

- A system prompt that defines the workspace zones, available action primitives (`pick`, `place`), and strict output formatting rules.
- A user message containing the current scene description (each detected object's ID, color, shape, and arm-space XY position) followed by the user's command.

The model is instructed to respond with only a valid JSON array. The response is stripped of any markdown fences before being parsed. The model runs at `temperature = 0` for fully deterministic planning.

The LLM used is `gemma4:e4b` running via Ollama. Token generation speed is monitored; a warning is printed if the model drops below 15 tokens/second, which indicates it may be running partially on CPU rather than GPU.

#### 5. Inverse Kinematics Solver

The `IKSolver` class implements analytical 2-link planar inverse kinematics for a RRRR configuration.

**Coordinate correction (`correct_arm_xy`):** Raw homography coordinates are passed through a piecewise-linear calibration polynomial before IK is solved. This accounts for mechanical imperfections in the physical arm such as non-ideal link lengths and servo mounting offsets.

**IK computation (`_compute_math_angles`):** The solver uses the following approach:

- Theta 1 (base rotation) is computed from the desired XY position, accounting for the end-effector's lateral offset (`EE_RIGHT`) using a two-argument arctangent.
- A virtual wrist position is computed by subtracting the end-effector forward projection (`EE_FORWARD`) from the radial distance.
- Theta 3 (elbow) is found using the law of cosines over link lengths L1 = 154 mm and L2 = 150 mm.
- Theta 2 (shoulder) is found from the remaining angle to the wrist point.
- Theta 4 (wrist) is set to keep the end-effector level: `theta4 = -(theta2 + theta3) - 5`.
- A z-axis dynamic compensation term `(200 - r) * 0.25` raises the elbow as the target approaches the arm base to avoid ground collisions.

Mathematical angles are then converted to servo angles by applying per-joint calibration offsets and direction signs stored in `SERVO_CAL`.

#### 6. Arm Controller and Motion Execution

The `ArmController` class manages the serial connection to the Arduino and orchestrates all physical motion.

**Auto-detection:** On startup, the class scans all available serial ports for known Arduino-compatible chip descriptions (CH340, CP210x, FTDI, ACM) and connects automatically.

**Transit interpolation (`_transit`):** When moving the arm horizontally between two XY positions, the path is divided into small segments (approximately 3 mm each). At each waypoint, IK is solved and a packet is sent. This prevents the arm from swinging through intermediate positions that could knock objects over or exceed joint limits mid-motion.

**Pick sequence (`pick`):**

1. Open gripper.
2. Move above the object at approach height (`Z_APPROACH = 0` mm).
3. Interpolated transit to object XY.
4. Lower to grip height (`Z_GRIP = -75` mm).
5. Close gripper.
6. Rise to a second intermediate height (`Z_GRIP2 = -50` mm).
7. Rise to approach height.

**Place sequence (`place`):**

1. Transit to destination XY at approach height.
2. Lower to grip height.
3. Open gripper.
4. Rise back to approach height.

**Main loop:** After each command is executed, the arm returns to photo-home so the workspace is clear for the next camera capture.

---

## Installation

**Python dependencies:**

```bash
pip install opencv-python numpy scipy sounddevice openai-whisper pyserial ollama
```

**Arduino dependencies:**

Install the Arduino IDE. The firmware uses only the built-in `Servo.h` library; no additional libraries are required.

**Local LLM:**

Install [Ollama](https://ollama.com) and pull the model used by the system:

```bash
ollama pull gemma4:e4b
```

Any Ollama-compatible model can be substituted by changing `LLM_MODEL` in `finalv1.py`.

---

## Setup and Usage

**Step 1 — Flash the firmware**

Open `IK1.ino` in the Arduino IDE, select the correct board and port, and upload. Open the Serial Monitor at 115200 baud to confirm the board is running.

**Step 2 — Calibrate the camera**

Place an A4 sheet flat on the workspace in the camera field of view. Run:

```bash
python a4_calibration.py --camera 1
```

Click the four corners of the sheet in the order shown on screen (bottom-left first), press Enter to compute the homography, verify the reprojection error is low (ideally under 2 mm), and press S to save `calibration.npz`. This file must be present in the same directory as `finalv1.py`.

**Step 3 — Run the controller**

```bash
python finalv1.py
```

The system will load Whisper, open the camera, connect to the Arduino, and move the arm to photo-home. When ready, it prompts for a command (press Enter to start recording if in voice mode).

**Example commands:**

- "Pick up the red circle and place it in the left zone."
- "Move the blue square to the right zone."
- "Place the object in the center zone."

Type `exit` or `quit` to shut down.

---

## Configuration Reference

All primary configuration variables are defined at the top of `finalv1.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_SRC` | `1` | Camera index or stream URL |
| `CAL_FILE` | `"calibration.npz"` | Path to saved homography |
| `CAMERA_WARMUP` | `30` | Frames discarded on camera open |
| `INPUT_MODE` | `"voice"` | `"voice"` or `"text"` |
| `WHISPER_MODEL` | `"base"` | Whisper model size |
| `LLM_MODEL` | `"gemma4:e4b"` | Ollama model name |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `CAPTURE_FRAMES` | `5` | Frames captured per scene scan |
| `CAPTURE_INTERVAL_S` | `0.15` | Seconds between frame grabs |
| `MATCH_RADIUS_MM` | `5.0` | Spatial tolerance for multi-frame fusion |
| `CONFIDENCE_THRESH` | `0.75` | Minimum color-fill ratio |
| `PHOTO_HOME_XYZ` | `(-50, -220, -50)` | Arm position during camera capture (mm) |
| `WORKSPACE_X` | `(0, 280)` | Valid arm X range (mm) |
| `WORKSPACE_Y` | `(-210, 210)` | Valid arm Y range (mm) |

IK constants (in `finalv1.py`):

| Constant | Value | Description |
|----------|-------|-------------|
| `L1` | 154.0 mm | Upper arm link length |
| `L2` | 150.0 mm | Forearm link length |
| `EE_FORWARD` | 160.0 mm | End-effector forward projection |
| `EE_RIGHT` | 30.0 mm | End-effector lateral offset |
| `Z_APPROACH` | 0.0 mm | Height for transit above objects |
| `Z_GRIP` | -75.0 mm | Height at which gripper closes |

---

## Acknowledgements

The 3D-printed arm model used in this project is from the open-source repository [doganhobby/Arduino-Projects](https://github.com/doganhobby/Arduino-Projects/tree/main/3D%20Robot%20Arm). Only the mechanical model was used; all software in this repository was written independently by the VoiceArm team.

Speech recognition is powered by [OpenAI Whisper](https://github.com/openai/whisper). LLM inference is provided by [Ollama](https://ollama.com). Computer vision uses [OpenCV](https://opencv.org).

---

## License

This project is released for academic and educational use. See `LICENSE` for details.
