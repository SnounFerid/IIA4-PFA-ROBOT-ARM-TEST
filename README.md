# VoiceArm — Natural Language Control of a 4-DOF Robotic Arm Using Local LLM Planning, Computer Vision, and Analytical Inverse Kinematics

> A fully offline, end-to-end embedded and AI pipeline that accepts spoken natural-language commands, understands a live scene through computer vision, plans manipulation actions using a locally-running large language model, and executes those actions on a 4-degree-of-freedom servo arm — with zero cloud dependency.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Team](#2-team)
3. [Robot](#3-robot)
4. [Motivation and Problem Statement](#4-motivation-and-problem-statement)
5. [System Architecture](#5-system-architecture)
   - 5.1 [High-Level Pipeline](#51-high-level-pipeline)
   - 5.2 [Data and Control Flow](#52-data-and-control-flow)
   - 5.3 [Module Responsibilities](#53-module-responsibilities)
6. [Hardware](#6-hardware)
   - 6.1 [Component List](#61-component-list)
   - 6.2 [Servo Pin Assignments](#62-servo-pin-assignments)
   - 6.3 [Physical Setup and Workspace](#63-physical-setup-and-workspace)
7. [Software Components](#7-software-components)
   - 7.1 [IK1.ino — Arduino Firmware](#71-ik1ino--arduino-firmware)
   - 7.2 [a4_calibration.py — Camera Calibration Tool](#72-a4_calibrationpy--camera-calibration-tool)
   - 7.3 [finalv1.py — Main Controller Pipeline](#73-finalv1py--main-controller-pipeline)
     - 7.3.1 [Camera Management](#731-camera-management)
     - 7.3.2 [Computer Vision Pipeline](#732-computer-vision-pipeline)
     - 7.3.3 [Speech Recognition](#733-speech-recognition)
     - 7.3.4 [LLM Task Planner](#734-llm-task-planner)
     - 7.3.5 [Inverse Kinematics Solver](#735-inverse-kinematics-solver)
     - 7.3.6 [Arm Controller and Motion Execution](#736-arm-controller-and-motion-execution)
     - 7.3.7 [Main Loop](#737-main-loop)
8. [Mathematical Foundations](#8-mathematical-foundations)
   - 8.1 [Homography and Perspective Transform](#81-homography-and-perspective-transform)
   - 8.2 [Inverse Kinematics Derivation](#82-inverse-kinematics-derivation)
   - 8.3 [Servo Calibration Mapping](#83-servo-calibration-mapping)
9. [Installation](#9-installation)
   - 9.1 [Prerequisites](#91-prerequisites)
   - 9.2 [Python Environment](#92-python-environment)
   - 9.3 [Arduino Firmware](#93-arduino-firmware)
   - 9.4 [Local LLM Setup](#94-local-llm-setup)
10. [Setup and Usage](#10-setup-and-usage)
11. [Configuration Reference](#11-configuration-reference)
    - 11.1 [System Configuration (finalv1.py)](#111-system-configuration-finalv1py)
    - 11.2 [IK and Geometry Constants](#112-ik-and-geometry-constants)
    - 11.3 [Arduino Firmware Parameters](#113-arduino-firmware-parameters)
12. [Design Decisions and Engineering Notes](#12-design-decisions-and-engineering-notes)
13. [Known Limitations and Future Work](#13-known-limitations-and-future-work)
14. [Troubleshooting](#14-troubleshooting)
15. [Repository Structure](#15-repository-structure)
16. [Acknowledgements](#16-acknowledgements)
17. [License](#17-license)

---

## 1. Abstract

VoiceArm is an academic robotics project that demonstrates the tight integration of five distinct technology domains — embedded systems, computer vision, speech recognition, large language model reasoning, and robot kinematics — into a single autonomous pipeline. The system allows a non-technical user to direct a 4-degree-of-freedom servo arm through spoken natural-language instructions such as "pick up the red circle and place it in the left zone," with no cloud dependency, no pre-programmed object positions, and no knowledge of coordinate systems required from the user.

At runtime, the system transcribes the user's voice using OpenAI Whisper, captures the workspace scene with an overhead camera and classifies objects by colour and shape using an HSV-based computer vision pipeline, passes a structured scene description to a locally-running Gemma 4 language model to generate a JSON action plan, solves the required joint angles for each motion target using analytical inverse kinematics, and transmits servo commands over USB serial to Arduino firmware that executes smooth, rate-limited motion.

The entire processing chain — from spoken word to physical motion — runs offline on a single host machine. The 3D-printed arm structure is adapted from an existing open-source model; all control software was designed and implemented independently by the project team.

---

## 2. Team

| Name |
|---|
| Snoun Ferid |
| Souheib Khelil |
| Elyes Grati |
| Oussema Njeh |

All team members contributed equally to the design, implementation, and testing of this project.

Institution: National Institute of Applied Sciences and Technology (INSAT), University of Carthage, Tunis, Tunisia

---

## 3. Robot

*A photograph of the assembled VoiceArm robot will be added here.*

<!-- Replace this comment with: ![VoiceArm assembled robot](images/robot.jpg) -->

---

## 4. Motivation and Problem Statement

Industrial robotic arms are typically programmed by entering precise numerical coordinates through a teach pendant or a dedicated programming environment. This creates a fundamental accessibility barrier: to move a robot arm to a specific location, the operator must know the target's coordinates in the robot's reference frame, understand the arm's joint configuration, and translate their intent into a sequence of low-level commands. This workflow is inaccessible to users without specialist robotics training.

The central question this project addresses is:

> Can a low-cost servo arm be commanded reliably through ordinary spoken language, using only locally-running AI components, without any cloud services or pre-programmed object locations?

This question has several sub-challenges that the system must address simultaneously:

- **Spatial grounding**: The system must be able to determine where objects physically are in the arm's coordinate frame without manual position entry.
- **Language understanding**: The system must interpret arbitrary natural-language phrasings of pick-and-place instructions and map them to structured robot actions.
- **Kinematic feasibility**: The system must translate target positions into physically realisable joint configurations, respecting the arm's geometry and joint limits.
- **Reliable execution**: The system must actuate servos smoothly enough to avoid damaging objects or the mechanism.
- **Privacy and robustness**: The system must operate entirely offline to avoid data privacy concerns and network dependency.

VoiceArm addresses all five of these challenges through the layered pipeline described in this document.

---

## 5. System Architecture

### 5.1 High-Level Pipeline

Each user command is processed through six sequential stages. The entire pipeline is triggered on every command and takes approximately 3 to 8 seconds from command submission to the arm beginning to move, depending on hardware.

```
+---------------------------------------------+
|         User Input (Voice or Text)           |
|   Press Enter -> speak command -> Enter      |
+---------------------------------------------+
                       |
                       v
+---------------------------------------------+
|         Speech Recognition                  |
|   OpenAI Whisper (base model, local)        |
|   Input:  16 kHz mono WAV audio             |
|   Output: plain text command string         |
+---------------------------------------------+
                       |
                       v
+---------------------------------------------+
|         Scene Capture                       |
|   Arm moves to photo-home (out of frame)    |
|   5 successive frames captured at 150 ms   |
|   intervals using overhead USB webcam       |
+---------------------------------------------+
                       |
                       v
+---------------------------------------------+
|         Computer Vision Pipeline            |
|   HSV colour segmentation                  |
|   Morphological noise filtering            |
|   Circularity-based shape classification   |
|   Dominant-colour verification             |
|   Multi-frame fusion (majority vote)       |
|   Homography coordinate transform          |
|   Output: list of {id, colour, shape, XY}  |
+---------------------------------------------+
                       |
                       v
+---------------------------------------------+
|         LLM Task Planner                    |
|   Model: Gemma 4 (gemma4:e4b) via Ollama   |
|   Input:  scene description + command      |
|   Output: JSON array of pick/place steps   |
+---------------------------------------------+
                       |
                       v
+---------------------------------------------+
|         IK Solver + Arm Controller          |
|   Coordinate correction (empirical)        |
|   Analytical 2R IK (law of cosines)        |
|   Transit interpolation (3 mm segments)    |
|   Pick / place motion sequences            |
|   Serial ASCII packet transmission         |
+---------------------------------------------+
                       |
                       v
+---------------------------------------------+
|         Arduino Firmware (IK1.ino)          |
|   Parse comma-delimited angle packets      |
|   Step-rate-limited servo motion           |
|   6 servos: base, shoulder x2, elbow,     |
|             wrist, gripper                 |
+---------------------------------------------+
```

### 5.2 Data and Control Flow

The following describes precisely what data passes between each stage and in what format:

- **Audio capture** produces a raw 16 kHz mono PCM NumPy array written to `command.wav`.
- **Whisper** reads the WAV file and returns a plain text string containing the transcribed command.
- **Camera capture** returns a Python list of BGR image frames as NumPy arrays of shape `(height, width, 3)`.
- **Vision pipeline** processes those frames and returns a list of detected-object dictionaries. Each dictionary contains: a unique string identifier (`obj_0`, `obj_1`, ...), a colour label, a shape label, pixel coordinates of the bounding-box centre, and arm-space XY coordinates in millimetres (`arm_xy`).
- **LLM planner** receives a formatted text block describing the scene and the user's command, and returns a parsed Python list of action-step dictionaries, each with keys `step`, `action` (`"pick"` or `"place"`), and `target` (an object ID or a named zone string).
- **Executor** maps each step to `arm.pick(x, y)` or `arm.place(x, y)` calls, retrieving millimetre coordinates from the detected object list or the zone definition dictionary.
- **IK solver** converts XYZ millimetre coordinates to four mathematical joint angles (theta1–theta4) in degrees, then applies per-joint calibration offsets to produce servo angles.
- **Serial layer** formats the four servo angles and the gripper state as a newline-terminated ASCII string and writes it to the USB serial port at 115200 baud.
- **Arduino firmware** parses the incoming string, updates its servo target array, and on each 10 ms firmware tick advances each servo one degree toward its target.

### 5.3 Module Responsibilities

| Module | Language | Primary Responsibility |
|---|---|---|
| `IK1.ino` | C++ (Arduino) | Parse serial angle packets; drive 6 servos with step-rate-limited motion |
| `a4_calibration.py` | Python 3 | Interactive tool: compute and save homography matrix from A4 sheet corners |
| `finalv1.py` | Python 3.11+ | Full pipeline integration: vision, STT, LLM, IK, serial control, main loop |

---

## 6. Hardware

### 6.1 Component List

| Component | Specification | Notes |
|---|---|---|
| Microcontroller | Arduino Uno (ATmega328P) or compatible | Any Arduino with 6 PWM pins is sufficient |
| Servos | 6x standard hobby servos | MG996R or equivalent torque-rated servos are recommended for the shoulder joints |
| Camera | USB webcam, 1280x720 at 30 fps | Must be mountable directly overhead, lens pointing straight down |
| Host computer | Any machine capable of running Python 3.11+ | A CUDA-capable GPU is strongly recommended for LLM inference |
| Calibration reference | A4 sheet of paper (210 x 297 mm) | Used only during calibration; standard printer paper is sufficient |
| 3D-printed frame | Arm model from doganhobby/Arduino-Projects | See Section 15 for attribution and source link |
| USB cable | Standard USB-A to USB-B | Arduino serial communication at 115200 baud |
| Power supply | Dedicated 5V / 6V supply for servos | Servos must NOT be powered from the Arduino 5V pin; insufficient current causes servo jitter and resets |

A GPU is strongly recommended for running the Gemma 4 language model at adequate speed. The system logs a warning if token generation falls below 15 tokens per second, which indicates the model has overflowed GPU VRAM and is executing partially on CPU. At that speed, the planning step alone takes several seconds per command.

### 6.2 Servo Pin Assignments

The Arduino firmware assigns each servo to a specific PWM-capable digital pin. These assignments are fixed in the firmware and must match the physical wiring of the assembled arm.

| Joint | Arduino Pin | Role |
|---|---|---|
| Shoulder 1 | 3 | Primary shoulder servo |
| Shoulder 2 | 5 | Mechanically mirrored: firmware sets this to `180 - shoulder1` automatically |
| Base | 6 | Horizontal base rotation |
| Elbow | 9 | Elbow flexion |
| Wrist | 10 | Held level by the IK constraint: `theta4 = -(theta2 + theta3) - 5` |
| Gripper | 11 | Boolean: 85 degrees = closed, 145 degrees = open |

### 6.3 Physical Setup and Workspace

The camera must be mounted directly above the workspace, with its optical axis perpendicular to the flat working surface. Any tilt introduces perspective distortion that the homography will partially correct, but severe tilt reduces calibration accuracy.

The A4 calibration sheet defines the origin of the camera coordinate system. When calibrating, the sheet's bottom-left corner becomes world origin (0, 0) mm. The arm should be positioned so that its reachable workspace — the area where IK produces valid solutions within joint limits — overlaps substantially with the camera's field of view.

The workspace bounds enforced by the software are:

| Axis | Minimum | Maximum | Description |
|---|---|---|---|
| X (arm forward) | 0 mm | 280 mm | Distance from arm base along the arm's forward axis |
| Y (arm lateral) | -210 mm | 210 mm | Lateral offset; negative is right, positive is left when facing the arm |

Objects detected outside these bounds are silently discarded before the scene description is built.

---

## 7. Software Components

### 7.1 IK1.ino — Arduino Firmware

**File:** `IK1.ino`
**Language:** C++ (Arduino framework)
**Purpose:** Receive joint angle commands over USB serial and drive six servo motors with smooth, rate-limited motion.

#### Overview

The firmware implements a target-tracking motion smoother. Rather than writing a received angle directly to each servo — which causes the arm to snap to its new position instantly, producing mechanical shock, audible clunking, and potential gear damage — the firmware separates each servo's *target* position from its *current* (live) position and closes the gap incrementally.

Two integer arrays are maintained in global scope:

- `cur[6]`: the current position of each servo, updated every tick.
- `tgt[6]`: the most recently received target position for each servo.

On every firmware tick (every `INTERVAL = 10` ms), the function `stepToward(current, target)` is called for each servo. It computes the difference between the current and target angles, clamps it to the range `[-STEP_SIZE, +STEP_SIZE]`, and advances the current position by that clamped step. With `STEP_SIZE = 1` degree and `INTERVAL = 10` ms, each servo moves at a maximum rate of 100 degrees per second, producing smooth, controlled motion regardless of how abruptly the host sends a new target.

The `loop()` function handles two tasks independently:

1. **Serial parsing**: If bytes are available on the serial port, it reads and parses one comma-delimited packet, updating the target array. This is non-blocking and does not interrupt servo stepping.
2. **Servo stepping**: Every `INTERVAL` milliseconds, all six servos are stepped toward their targets and their new angles are written to the physical servos.

#### Serial Protocol

The host Python code sends newline-terminated ASCII strings at 115200 baud. Each string contains five comma-separated values:

```
<base>,<shoulder>,<elbow>,<wrist>,<gripper>\n
```

All angle values are decimal integers representing servo pulse-width angles in degrees. The gripper field is `0` (open) or `1` (close). Example:

```
95,60,90,33,0
```

The firmware parser reads the string using `Serial.readStringUntil('\n')` and iterates through commas with `indexOf()` and `substring()`. The `Serial.setTimeout(10)` call in `setup()` ensures the read does not block indefinitely if a packet is malformed.

#### Shoulder Mirroring

The physical arm uses two servos mounted on opposing sides of the shoulder joint, both driving the same mechanical joint. When one servo rotates clockwise, the other must rotate counter-clockwise by the same amount to produce a symmetric force. The firmware handles this automatically:

```cpp
tgt[1] = val[1];           // shoulder1: received angle
tgt[2] = 180 - val[1];    // shoulder2: mirrored
```

The host only sends one shoulder angle. The mirroring is invisible to the Python code.

#### Gripper Logic

The gripper is deliberately driven by a boolean target rather than a continuous angle, to prevent the host from commanding an intermediate position that might crush objects or leave the gripper partly closed during transit. When the gripper field is `1`, the target is set to `85` (closed); when `0`, it is set to `145` (open). These positions are specific to the physical gripper geometry and may need adjustment for different gripper designs.

#### Firmware Parameters

| Parameter | Default | Effect |
|---|---|---|
| `STEP_SIZE` | `1` degree | Maximum angular change per tick per servo. Increase for faster motion, decrease for smoother motion. |
| `INTERVAL` | `10` ms | Tick period. Smaller values increase update frequency but leave less CPU time for serial parsing. |
| Serial baud rate | `115200` | Must exactly match `ArmController.BAUD_RATE` in `finalv1.py`. |
| Gripper closed angle | `85` | Adjust if the gripper does not fully close on the physical hardware. |
| Gripper open angle | `145` | Adjust if the gripper does not fully open on the physical hardware. |

---

### 7.2 a4_calibration.py — Camera Calibration Tool

**File:** `a4_calibration.py`
**Language:** Python 3
**Dependencies:** `opencv-python`, `numpy`
**Purpose:** Compute and persistently save the homography matrix that maps camera pixel coordinates to real-world millimetre coordinates, using a standard A4 sheet as the ground-truth reference.

#### The Calibration Problem

The camera observes the workspace from above and produces a 2D image in pixel coordinates. The robot arm, however, must be commanded using physical millimetre coordinates in its own reference frame. Bridging this gap requires a spatial mapping from image space to world space.

If the camera were mounted perfectly perpendicular to a perfectly flat workspace, a simple uniform scale factor would suffice. In practice, however, the camera is slightly tilted, the lens introduces barrel or pincushion distortion, and the workspace may not be perfectly level relative to the camera sensor. A perspective homography transformation accounts for all of these effects simultaneously. It is a projective transformation that maps any 2D point in the image to its corresponding 2D position on the flat workspace plane, handling scale differences, rotation, shear, and perspective foreshortening in a single 3x3 matrix operation.

#### Why A4 Paper?

An A4 sheet is an ideal calibration target for this application because its dimensions (210 x 297 mm) are defined by an international standard and are known to sub-millimetre accuracy. It is flat, has high contrast black text or borders, is available anywhere, and requires no printing or preparation — only placement. Four corners are the minimum number of correspondences needed to uniquely determine a homography, and a sheet provides exactly four well-defined, unambiguous corners.

#### Calibration Procedure

The user places an A4 sheet flat within the camera's field of view and runs the tool. An interactive OpenCV window opens showing the live camera feed. The user clicks the four corners of the sheet in a fixed, documented order. Each click registers a pixel coordinate that is paired with its known real-world millimetre coordinate:

| Click order | Corner | World coordinates (mm) |
|---|---|---|
| 1 | Bottom-left | (0, 0) |
| 2 | Bottom-right | (210, 0) |
| 3 | Top-right | (210, 297) |
| 4 | Top-left | (0, 297) |

Once all four corners have been placed, the tool calls `cv2.findHomography()` to compute the 3x3 perspective transformation matrix `H` using the Direct Linear Transform (DLT) algorithm. The mean reprojection error — the average distance in millimetres between each corner's projected position and its expected world position — is computed and reported. A reprojection error below 2 mm indicates good calibration; above 5 mm indicates imprecise corner clicking and the calibration should be redone.

#### Resolution Locking

A critical design requirement implemented in this tool is that the camera resolution at calibration time is stored inside `calibration.npz` alongside the matrix `H`. Homography matrices are resolution-dependent: the same physical point maps to a different pixel coordinate at a different resolution, making the stored matrix invalid if the resolution is changed between calibration and runtime. When `finalv1.py` loads the calibration file, it reads the stored resolution and compares it to the current camera resolution. If they differ, a visible warning is printed because all coordinate mappings will be systematically incorrect until the camera is recalibrated at the new resolution.

#### Interactive Controls

| Action | Mode | Effect |
|---|---|---|
| Left-click | Corner selection | Place the next corner at the clicked pixel |
| Right-click | Corner selection | Undo and remove the last placed corner |
| Enter | Corner selection | Compute homography from the four corners and switch to measure mode |
| Left-click | Measure | Print the millimetre coordinate of the clicked pixel to the terminal |
| S | Measure | Save `H`, pixel corners, and resolution to `calibration.npz` |
| L | Any | Load a previously saved calibration from `calibration.npz` |
| R | Any | Discard all corners and return to corner selection mode |
| Esc | Any | Exit the tool |

#### Command-Line Arguments

```bash
# Default: live webcam at index 1, resolution 1280x720
python a4_calibration.py

# Use a static photograph instead of a live webcam
python a4_calibration.py --image photo.jpg

# Specify a different camera device index
python a4_calibration.py --camera 0

# Specify a non-default capture resolution
python a4_calibration.py --width 1920 --height 1080
```

#### Headless API

Three standalone functions are exported by this module for use by other scripts that require coordinate conversion without running the interactive UI:

```python
# Load the homography matrix from a saved calibration file
H = load_homography("calibration.npz")

# Convert a pixel coordinate (px, py) to arm-space millimetres
x_mm, y_mm = pixel_to_mm(px, py, H)

# Convert arm-space millimetres back to pixel coordinates (uses H inverse)
px, py = mm_to_pixel(x_mm, y_mm, H)
```

---

### 7.3 finalv1.py — Main Controller Pipeline

**File:** `finalv1.py`
**Language:** Python 3.11+
**Dependencies:** `opencv-python`, `numpy`, `scipy`, `sounddevice`, `openai-whisper`, `pyserial`, `ollama`
**Purpose:** Full end-to-end pipeline integrating camera management, HSV-based computer vision, local speech recognition, LLM task planning, analytical inverse kinematics, and serial arm control.

---

#### 7.3.1 Camera Management

The camera is opened once at startup using `cv2.VideoCapture` and held open for the entire runtime session. Opening and closing the camera device on every command would introduce a multi-second hardware initialisation delay. Immediately after opening, `CAMERA_WARMUP = 30` frames are captured and discarded, because most USB webcams require several frames of exposure settling before auto-exposure and auto-white-balance stabilise. Using a frame from this warm-up period would produce unreliable colour measurements.

Before every scene scan, the function `camera_capture(arm)` calls `arm.move_to_photo_home()` to drive the arm to the fixed position `PHOTO_HOME_XYZ = (-50, -220, -50)` mm — a location that clears the entire workspace surface from the camera's field of view. Only after the arm has reached this position does the function begin acquiring frames. It captures `CAPTURE_FRAMES = 5` frames with a `CAPTURE_INTERVAL_S = 0.15` second pause between each, producing a total capture window of approximately 0.6 seconds per scene scan.

---

#### 7.3.2 Computer Vision Pipeline

The vision pipeline is built around robustness to real-world indoor lighting variation. It operates entirely in HSV (Hue, Saturation, Value) colour space rather than RGB, because HSV decouples the chromatic identity of a colour (hue) from its brightness (value) and colourfulness (saturation). This means that a red object under dim lighting and a red object under bright lighting share a similar hue value, even though their RGB representations may be completely different. HSV-based thresholds are therefore far more stable across real-world illumination changes than RGB thresholds.

**Colour Mask Construction**

The pipeline defines HSV range pairs for six colours: red, orange, yellow, green, blue, and black. Red requires special treatment because the hue axis in OpenCV wraps from 180 back to 0, meaning a pure red can appear near hue 0 or near hue 180. Two separate HSV ranges are defined for red and their masks combined with a bitwise OR operation.

After range-based masking, each mask undergoes two morphological operations using a 5x5 elliptical structuring element:
- **Morphological opening** (erosion followed by dilation): eliminates isolated noise pixels and small false-positive patches that do not correspond to real objects.
- **Morphological closing** (dilation followed by erosion): fills small holes and gaps within genuine object regions that arise from specular highlights or surface texture.

**Shape Classification**

For each contour found in a colour mask (above a minimum area threshold of 600 pixels), the function `classify_shape()` computes the contour's **circularity**, defined as:

```
circularity = (4 * pi * area) / perimeter^2
```

A geometrically perfect circle has circularity equal to 1.0; rectangles and irregular shapes score progressively lower. The following classification rules are applied:

| Condition | Assigned shape |
|---|---|
| Circularity > 0.85 | circle |
| Exactly 4 vertices after polygon approximation (epsilon = 3% of perimeter) and aspect ratio between 0.85 and 1.15 | square |
| Exactly 4 vertices and aspect ratio between 0.25 and 0.85, or between 1.15 and 4.0 | rectangle |
| More than 6 vertices and circularity > 0.65 | circle (handles slightly blurred or irregular physical circles) |
| Any other case | rejected (not returned as a detection) |

Contours whose aspect ratio falls outside [0.25, 4.0] are rejected as probable noise, thin lines, or frame edges.

**Colour Confirmation**

After shape classification, a secondary colour verification step is applied to every accepted contour. The function `dominant_color_in_contour()` draws the interior of the contour as a binary mask, then counts how many pixels inside that region match each of the six supported colour HSV ranges. The colour with the highest inlying pixel count is provisionally accepted as the object's colour, subject to a minimum fill threshold:

| Colour | Minimum fill threshold |
|---|---|
| Red, orange, yellow, green, blue | 20% of contour area |
| Black | 10% of contour area (lower due to black's near-zero saturation reducing mask coverage) |

This double-check prevents a contour found in one colour's mask from being labelled with that colour due to bleed between adjacent HSV ranges.

**Multi-Frame Fusion**

Single-frame detection is inherently susceptible to transient noise: a brief specular reflection, a shadow from hand movement, or a small camera vibration can cause an object to be missed or misclassified in an individual frame. The system addresses this through temporal fusion across the five captured frames.

The fusion algorithm works as follows: after processing all frames independently, detections from different frames are compared pairwise. Two detections are considered to represent the same physical object if the distance between their arm-space XY centres is within `MATCH_RADIUS_MM = 5.0` mm. Matched detections are grouped together, and for each group:
- The final **shape** label is determined by majority vote across all frames in the group.
- The final **colour** label is determined by majority vote across all frames in the group.
- The final **position** is the centroid of all matched positions in the group.

This means that a single-frame misclassification — for example, a yellow object classified as orange in one of five frames due to a momentary light flicker — does not propagate to the final scene description, as long as at least three frames classify it correctly.

**Coordinate Transform**

Each fused detection's arm-space XY position is computed by passing its pixel-space centroid `(cx, cy)` through `pixel_to_mm()`. This function applies the homography matrix `H` using OpenCV's `cv2.perspectiveTransform()`. After the transformation, the two output axes are deliberately swapped:

```python
arm_x = result[0][0][1]   # raw homography row-axis  -> arm X (forward)
arm_y = result[0][0][0]   # raw homography col-axis  -> arm Y (lateral)
```

This axis swap was determined empirically during system integration to align the camera coordinate frame with the physical arm coordinate frame, given the specific mounting orientation of the camera and the placement convention of the A4 calibration sheet.

---

#### 7.3.3 Speech Recognition

When `INPUT_MODE = "voice"`, the system uses OpenAI Whisper running entirely on the local machine — no audio data is sent to any external service. The recording flow is as follows:

1. The main loop prompts the user by printing a message and waiting for Enter.
2. On the first Enter, a background thread starts audio recording using `sounddevice`. The background thread fills a list of audio chunks continuously, each chunk covering 100 ms of audio at 16 kHz.
3. The main thread waits for a second Enter press without blocking the recording thread.
4. On the second Enter, a thread event signals the recording thread to stop. The main thread joins the recording thread and waits for any in-progress chunk to complete.
5. All captured chunks are concatenated into a single NumPy array, written to `command.wav` as a 16-bit PCM WAV file, and passed to `whisper_model.transcribe()`.
6. The transcribed text string is returned as the command.

The Whisper `"base"` model (74 million parameters) is selected as the default because it provides adequate transcription accuracy for short commands while remaining practical for CPU execution. Larger models (`"small"` at 244M, `"medium"` at 769M, `"large"` at 1550M parameters) are available via the `WHISPER_MODEL` constant and may improve accuracy for users with heavy accents or noisy environments.

When `INPUT_MODE = "text"`, the system bypasses audio capture entirely and reads the command from standard input, which is useful for testing the vision and planning pipeline without a microphone.

---

#### 7.3.4 LLM Task Planner

The task planner is the component that bridges natural language and structured robot commands. Rather than implementing a hand-crafted parser — which would need to enumerate every possible phrasing a user might employ — the system delegates linguistic interpretation to a locally-running large language model that handles arbitrary natural-language variation inherently.

The planner is invoked via the Ollama Python library, which provides a local API to the Gemma 4 model. It sends two messages:

**System Prompt**

The system prompt defines the robot's operational context precisely and constrains the model's output format strictly. It specifies:
- The named zones available on the workspace (`left_zone`, `right_zone`, `center_zone`) and their approximate arm-space coordinates.
- The two action primitives available: `pick(obj_id)` selects an object to lift; `place(target_id)` deposits the held object at a zone or beside another object.
- Hard constraints: never hold two objects simultaneously; always pick before placing unless specifically instructed otherwise; use only valid object IDs or zone names as targets; output exclusively a valid JSON array with no surrounding text, explanation, or markdown.

**User Message**

The user message contains a text-formatted snapshot of the current scene followed by the user's command:

```
Scene:
  obj_0: red circle at arm_xy=(120.0, -30.0) mm
  obj_1: blue square at arm_xy=(200.0, 45.0) mm
  obj_2: green rectangle at arm_xy=(150.0, 110.0) mm

Command: "pick up the blue square and put it on the left"
```

**Model Response**

The model is invoked with `temperature = 0`, ensuring fully deterministic output for reproducible behaviour. A valid response from the model looks like:

```json
[
  {"step": 1, "action": "pick",  "target": "obj_1"},
  {"step": 2, "action": "place", "target": "left_zone"}
]
```

The response is stripped of any markdown code fences (` ```json `) before JSON parsing, accommodating models that add formatting despite being instructed not to.

The model used is `gemma4:e4b` — Gemma 4 in its `e4b` (approximately 4 billion effective parameters) quantisation, a balanced choice of instruction-following quality versus inference speed on consumer-grade GPU hardware. Token generation speed is measured at every call; values below 15 tokens/second trigger a warning suggesting VRAM overflow to CPU.

---

#### 7.3.5 Inverse Kinematics Solver

The `IKSolver` class implements a closed-form analytical inverse kinematics solution for the arm. The arm is modelled as a 2R planar manipulator (two links of lengths L1 = 154 mm and L2 = 150 mm) operating in a vertical plane, preceded by a rotational base joint (theta1) that rotates that plane around the vertical axis, and followed by a wrist joint (theta4) that compensates for the combined shoulder-elbow inclination to keep the end-effector horizontal.

The end-effector introduces two additional offsets that must be accounted for in the IK: a forward projection `EE_FORWARD = 160 mm` along the arm plane, and a lateral offset `EE_RIGHT = 30 mm` perpendicular to the arm plane (due to the gripper's physical geometry).

**Empirical Coordinate Correction**

Before the IK equations are applied, raw arm-space coordinates are passed through `correct_arm_xy()`, a piecewise-linear function that compensates for systematic mechanical errors that arise from imperfect link lengths, servo mounting offsets, and gearbox backlash:

```python
def correct_arm_xy(px, py):
    x = 1.0913 * px - 8.7152 + 108        # linear X correction
    if py <= -50:
        y = 1.0  * py - 12                 # right-side Y correction
    elif py >= 50:
        y = 1.22 * py - 5                  # left-side Y correction
    else:
        y = 1.1  * py + 0                  # central Y correction
    return x, y
```

The piecewise Y correction reflects the observation that the arm's effective lateral positioning error varies across the workspace in a non-linear way that can be approximately modelled as three separate linear regions. The correction coefficients were determined empirically by commanding the arm to known positions and measuring physical end-effector displacement.

**Analytical IK Computation**

The IK solution is computed by `_compute_math_angles()` in the following sequence:

1. **Base angle (theta1):** The base must rotate to align the arm plane with the target position, corrected for the end-effector's lateral offset. Using the two-argument arctangent:

   ```
   A        = sqrt(x^2 + y^2 - EE_RIGHT^2)
   theta1   = atan2(y, x) - atan2(EE_RIGHT, A)
   ```

2. **Virtual wrist position (rw):** The forward end-effector projection is subtracted from the total reach along the arm plane to find the distance from the shoulder to the wrist joint:

   ```
   rw = A - EE_FORWARD
   ```

3. **Dynamic Z compensation:** A term is added to the target Z to progressively raise the elbow as the target radius decreases, preventing the elbow from colliding with the workspace surface during short-reach motions:

   ```
   z_compensated = z + (200 - r) * 0.25
   ```

4. **Elbow angle (theta3):** Determined by the law of cosines applied to the triangle formed by the upper arm (L1), the forearm (L2), and the straight-line distance from shoulder to wrist:

   ```
   dist^2     = rw^2 + z^2
   cos(theta3) = (dist^2 - L1^2 - L2^2) / (2 * L1 * L2)
   theta3      = acos(clamped(cos(theta3), -1.0, 1.0))
   ```

   The cosine argument is clamped to [-1, 1] before the inverse cosine to handle numerical rounding near the workspace boundary, which could otherwise produce domain errors. The sign of theta3 is determined by the `elbow_up` flag.

5. **Shoulder angle (theta2):** Computed from the remaining angle in the shoulder-wrist-target triangle:

   ```
   theta2 = atan2(z, rw) - atan2(L2 * sin(theta3), L1 + L2 * cos(theta3))
   ```

6. **Wrist level correction (theta4):** Set to maintain a horizontal end-effector by compensating for the combined angular displacement introduced by the shoulder and elbow:

   ```
   theta4 = -(theta2 + theta3) - 5
   ```

   The constant -5 is a physical calibration offset specific to the wrist servo zero position on the assembled arm.

**Joint Limit Checking**

Before returning, `_check_limits()` verifies that all four mathematical joint angles fall within the physically reachable ranges defined in `MATH_LIMITS`. If any joint limit is violated, a `ValueError` is raised with a diagnostic message specifying the joint, its requested angle, and its allowed range. The executor catches this exception and aborts the current plan step.

---

#### 7.3.6 Arm Controller and Motion Execution

The `ArmController` class manages the USB serial connection to the Arduino and orchestrates all physical motion sequences. It is used as a Python context manager (`with ArmController() as arm:`), which guarantees that the serial port is properly closed even if an exception occurs.

**Serial Port Auto-Detection**

On instantiation, the class scans all available system serial ports using `serial.tools.list_ports.comports()` and matches each port's description against a list of known USB-to-serial chip identifiers: `arduino`, `ch340`, `cp210`, `ftdi`, `acm`, and `usb serial`. The first matching port is selected automatically. If no matching port is found, the exception message lists all available ports so the user can diagnose wiring or driver issues and supply the correct port manually.

**Transit Interpolation**

When the arm must move horizontally between two XY positions — for example, from the pick location to the place zone — simply solving IK for the destination and transmitting a single packet would allow the arm to take any path through the workspace. Intermediate configurations could knock nearby objects over, swing through singularities, or briefly place joints outside their safe range of motion.

The `_transit()` method prevents this by dividing the straight-line path between the current and target XY positions into small segments of approximately 3 mm each. For each intermediate waypoint, IK is solved and a serial packet is sent. Packets are sent at 15 ms intervals, and the firmware's rate limiter ensures smooth following even at this rate. Waypoints where IK raises a `ValueError` (outside the workspace boundary) are silently skipped, which handles the case where the straight-line path passes close to the arm base.

**Pick Sequence**

A complete pick operation for an object at `(x, y)` executes the following fixed sequence:

| Step | Action | Z height |
|---|---|---|
| 1 | Open gripper | — |
| 2 | Move above object | Z_APPROACH = 0 mm |
| 3 | Interpolated transit to (x, y) | Z_APPROACH = 0 mm |
| 4 | Lower to grip position | Z_GRIP = -75 mm |
| 5 | Wait MOVE_DELAY = 1.2 s for arm to settle | — |
| 6 | Close gripper; wait 0.6 s | — |
| 7 | Rise to intermediate height | Z_GRIP2 = -50 mm |
| 8 | Rise to approach height | Z_APPROACH = 0 mm |

**Place Sequence**

A complete place operation for a destination at `(x, y)` executes:

| Step | Action | Z height |
|---|---|---|
| 1 | Interpolated transit to destination (x, y) | Z_APPROACH = 0 mm |
| 2 | Lower to release position | Z_GRIP = -75 mm |
| 3 | Open gripper; wait 0.6 s | — |
| 4 | Rise to approach height | Z_APPROACH = 0 mm |

After every complete command execution, the arm returns to `PHOTO_HOME_XYZ` so the workspace is always clear before the next camera capture.

---

#### 7.3.7 Main Loop

The main application loop in `main()` executes the following sequence on every iteration:

1. Call `get_user_command()` — either prompt voice recording via Whisper or read from standard input.
2. If the command is `"exit"` or `"quit"`, break the loop and close all resources.
3. If the command is fewer than 3 characters (accidental Enter press or transcription artefact), skip and re-prompt.
4. Call `camera_capture(arm)` — move to photo-home and acquire the multi-frame scene buffer.
5. Call `opencv_detect(frames)` — run the full vision pipeline and return a list of detected objects.
6. If the detected list is empty, print a message and return to step 1 without calling the LLM.
7. Call `build_scene_description(detected_objects)` — format the object list as a human-readable text block.
8. Call `call_llm_planner(scene_description, command)` — send the scene and command to the LLM and parse the JSON response.
9. Call `execute_plan(plan, detected_objects, arm)` — execute each pick/place step against the arm hardware.
10. Call `arm.move_to_photo_home()` — return the arm to the neutral position.

Each of steps 4, 7, 8, and 9 is wrapped in exception handling. Camera failures, vision pipeline errors, LLM JSON parse failures, and IK `ValueError` exceptions are all caught, logged with a descriptive message, and the loop continues to the next iteration rather than crashing the session.

---

## 8. Mathematical Foundations

### 8.1 Homography and Perspective Transform

A homography is a 3x3 matrix `H` that encodes a projective transformation between two planes. In this system, the two planes are the image plane (measured in pixels) and the workspace plane (measured in millimetres).

Given a point `p = [u, v, 1]^T` in homogeneous image coordinates, the corresponding homogeneous world coordinate `p' = [x', y', w]^T` is computed as:

```
[x']       [u]
[y'] = H * [v]
[w ]       [1]
```

The actual world coordinates in millimetres are recovered by dividing by the homogeneous scale factor:

```
x_mm = x' / w
y_mm = y' / w
```

OpenCV's `cv2.findHomography()` computes `H` using the Direct Linear Transform (DLT) algorithm from the four corner point correspondences. With exactly four correspondences and no measurement noise, the system of equations is exactly determined and the DLT produces an exact solution. In practice, small errors in corner clicking introduce slight reprojection error, which is why the calibration tool reports this metric.

The homography is valid only on the flat workspace plane at the height at which calibration was performed. Objects resting at a different height (e.g. stacked on top of one another) will have their positions mapped with a systematic error. This is an inherent limitation of planar homography and is listed in Section 12.

### 8.2 Inverse Kinematics Derivation

The arm is modelled as a **2R planar manipulator** operating in a vertical plane (the arm plane), which is rotated around the vertical Z axis by the base joint (theta1). The end-effector is displaced from the wrist joint by a forward offset and a lateral offset, making the effective wrist position different from the raw target position.

**Base angle (theta1)**

The base must rotate to a corrected angle to align the arm plane with the target, accounting for the lateral end-effector offset `EE_RIGHT`:

```
A      = sqrt(x^2 + y^2 - EE_RIGHT^2)    (radial reach in the arm plane)
theta1 = atan2(y, x) - atan2(EE_RIGHT, A)
```

**Virtual wrist reach (rw)**

Subtracting the forward end-effector projection from the arm-plane reach gives the wrist position:

```
rw = A - EE_FORWARD
```

**Elbow angle (theta3) by law of cosines**

Let `dist = sqrt(rw^2 + z^2)` be the straight-line distance from the shoulder to the wrist. By the law of cosines applied to the triangle formed by L1, L2, and dist:

```
cos(theta3) = (dist^2 - L1^2 - L2^2) / (2 * L1 * L2)
theta3      = acos(cos(theta3))           (elbow-up: positive; elbow-down: negative)
```

**Shoulder angle (theta2)**

Using the two-argument arctangent to find the elevation angle to the wrist, then subtracting the angle introduced by the forearm:

```
theta2 = atan2(z, rw) - atan2(L2 * sin(theta3), L1 + L2 * cos(theta3))
```

**Wrist compensation (theta4)**

The wrist joint is set to keep the end-effector horizontal at all arm configurations. The total downward inclination of the end-effector equals the sum of the shoulder and elbow angles. Negating this sum compensates for it:

```
theta4 = -(theta2 + theta3) - 5
```

The constant -5 degrees is a measured physical offset for the wrist servo's zero-degree position on the assembled hardware.

### 8.3 Servo Calibration Mapping

Mathematical joint angles (in degrees, measured in the IK geometric frame) do not correspond directly to servo pulse-width angles. Each servo has its own zero-point offset, direction of positive rotation, and reference frame. The conversion is applied using per-joint calibration tuples stored in `SERVO_CAL`:

| Joint | Servo offset | Math offset | Direction |
|---|---|---|---|
| theta1 (base) | 95.0 | 0.0 | +1 |
| theta2 (shoulder) | 155.0 | 90.0 | +1 |
| theta3 (elbow) | 169.0 | 0.0 | +1 |
| theta4 (wrist) | 33.0 | 0.0 | +1 |

The conversion formula for each joint is:

```
servo_angle = servo_offset + direction_sign * (math_angle - math_offset)
```

These constants were determined by physically moving the arm to known reference configurations (e.g. arm pointing straight forward, elbow at 90 degrees) and recording the servo angles at which those configurations were achieved.

---

## 9. Installation

### 9.1 Prerequisites

- Python 3.11 or later (required for type hint syntax used in `finalv1.py`)
- Arduino IDE 1.8 or 2.x
- [Ollama](https://ollama.com) installed and running as a background service
- A CUDA-capable GPU with at least 6 GB VRAM is recommended for LLM inference at usable speed. The system functions on CPU only but planning latency may be 30–60 seconds per command.

### 9.2 Python Environment

It is strongly recommended to use a dedicated virtual environment to isolate dependencies:

```bash
python -m venv voicearm-env

# Activate on Linux / macOS
source voicearm-env/bin/activate

# Activate on Windows (Command Prompt)
voicearm-env\Scripts\activate.bat

# Activate on Windows (PowerShell)
voicearm-env\Scripts\Activate.ps1
```

Install all required Python packages:

```bash
pip install opencv-python numpy scipy sounddevice openai-whisper pyserial ollama
```

Verify the installation by importing all dependencies:

```bash
python -c "import cv2, numpy, scipy, sounddevice, whisper, serial, ollama; print('All dependencies OK')"
```

### 9.3 Arduino Firmware

1. Open the Arduino IDE.
2. Open `IK1.ino` via **File > Open**.
3. Under **Tools > Board**, select the correct Arduino model (e.g. Arduino Uno).
4. Under **Tools > Port**, select the USB serial port that corresponds to the connected Arduino.
5. Click the **Upload** button (right arrow icon) and wait for "Done uploading."
6. Open **Tools > Serial Monitor**, set the baud rate dropdown to `115200`, and confirm the board is running. The firmware will be silent until it receives a serial packet — this is the expected behaviour.

### 9.4 Local LLM Setup

Install Ollama from [https://ollama.com](https://ollama.com) following the instructions for your operating system. Then pull the required model:

```bash
ollama pull gemma4:e4b
```

Depending on network speed, this download may take several minutes. Verify the model runs correctly:

```bash
ollama run gemma4:e4b "Reply with the single word: ready"
```

The model should respond with `ready`. If it does not, check that the Ollama background service is running.

Any Ollama-compatible instruction-tuned model can be substituted by changing the `LLM_MODEL` constant in `finalv1.py`. Models with stronger JSON instruction-following — such as `llama3`, `mistral`, or `phi3` — may produce more reliable output on complex multi-step commands.

---

## 10. Setup and Usage

### Step 1 — Assemble and wire the hardware

Wire each servo signal wire to the correct Arduino digital pin as listed in the pin assignment table in Section 5.2. Connect servo power and ground to a dedicated 5V or 6V power supply, not to the Arduino's 5V output (insufficient current causes servo jitter and can reset the Arduino). Connect the Arduino to the host computer via USB.

Mount the camera overhead, as close to perpendicular to the workspace surface as possible. The camera should have a clear view of the entire working area.

### Step 2 — Flash the firmware

Upload `IK1.ino` to the Arduino as described in Section 8.3. Confirm the board is running by opening the Serial Monitor at 115200 baud.

### Step 3 — Calibrate the camera

Place an A4 sheet flat on the workspace surface within the camera's field of view. Run the calibration tool:

```bash
python a4_calibration.py --camera 1
```

Follow the on-screen instructions:
- Click the four corners of the A4 sheet in the order shown: **bottom-left, bottom-right, top-right, top-left**.
- Press **Enter** to compute the homography.
- Inspect the reported reprojection error. Values below 2 mm are good; values above 5 mm indicate imprecise corner clicking and calibration should be repeated with **R**.
- Press **S** to save the calibration to `calibration.npz` in the current working directory.

The file `calibration.npz` must be present in the same directory as `finalv1.py` when the main controller is started.

### Step 4 — Run the main controller

```bash
python finalv1.py
```

On startup, the system will:
1. Print the active input mode (`VOICE` or `TEXT`).
2. Load the Whisper speech recognition model (may take 10–30 seconds on first run).
3. Open the camera and discard warm-up frames.
4. Scan for and connect to the Arduino serial port automatically.
5. Move the arm to photo-home.
6. Print `System ready.` and prompt for the first command.

### Step 5 — Issue commands

**In voice mode** (`INPUT_MODE = "voice"`):
1. Press Enter to begin recording.
2. Speak your command clearly.
3. Press Enter again to stop recording and submit.

**In text mode** (`INPUT_MODE = "text"`):
1. Type the command at the prompt and press Enter.

**Representative command examples:**

```
Pick up the red circle and place it in the left zone.
Move the blue square to the right.
Put the yellow object in the center.
Place the green rectangle on the left.
Pick up everything and sort it to the right.
```

To end the session cleanly, type or say `exit` or `quit`.

---

## 11. Configuration Reference

### 11.1 System Configuration (finalv1.py)

All configuration variables are defined as module-level constants at the top of `finalv1.py`. They can be modified without changing any logic code.

| Variable | Default | Description |
|---|---|---|
| `CAMERA_SRC` | `1` | Camera device index (integer) or RTSP/HTTP stream URL (string). Use `0` if only one camera is connected. |
| `CAL_FILE` | `"calibration.npz"` | Path to the homography calibration file produced by `a4_calibration.py`. |
| `CAMERA_WARMUP` | `30` | Number of frames discarded after camera open to allow auto-exposure to stabilise. |
| `INPUT_MODE` | `"voice"` | Command input mode: `"voice"` activates Whisper STT; `"text"` reads from keyboard. |
| `WHISPER_MODEL` | `"base"` | Whisper model size. Options: `"tiny"` (39M), `"base"` (74M), `"small"` (244M), `"medium"` (769M), `"large"` (1550M). |
| `LLM_MODEL` | `"gemma4:e4b"` | Ollama model identifier. Any instruction-tuned model available in Ollama is accepted. |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz. Whisper is trained on 16 kHz audio; do not change this. |
| `CAPTURE_FRAMES` | `5` | Number of successive frames captured per scene scan. Higher values improve fusion reliability at the cost of capture time. |
| `CAPTURE_INTERVAL_S` | `0.15` | Pause in seconds between successive frame captures. |
| `MATCH_RADIUS_MM` | `5.0` | Maximum arm-space distance (mm) between detections in different frames for them to be fused as the same object. |
| `CONFIDENCE_THRESH` | `0.75` | Minimum colour fill ratio for a detection to pass the colour confirmation step. |
| `PHOTO_HOME_XYZ` | `(-50, -220, -50)` | Arm position (mm, raw before `correct_arm_xy`) used during camera capture. Must be outside the camera field of view. |
| `WORKSPACE_X` | `(0, 280)` | Valid arm X coordinate range in mm. Detections outside this range are discarded. |
| `WORKSPACE_Y` | `(-210, 210)` | Valid arm Y coordinate range in mm. Detections outside this range are discarded. |

### 11.2 IK and Geometry Constants

| Constant | Value | Description |
|---|---|---|
| `L1` | 154.0 mm | Upper arm link length, measured from shoulder servo axis to elbow servo axis. |
| `L2` | 150.0 mm | Forearm link length, measured from elbow servo axis to wrist servo axis. |
| `EE_FORWARD` | 160.0 mm | End-effector forward projection from the wrist axis to the gripper tip. |
| `EE_RIGHT` | 30.0 mm | End-effector lateral offset from the arm plane to the gripper centreline. |
| `Z_APPROACH` | 0.0 mm | Z height used for horizontal transit. The arm clears all objects at this height. |
| `Z_GRIP` | -75.0 mm | Z height at which the gripper closes around an object. Adjust if the gripper does not contact the surface. |
| `Z_GRIP2` | -50.0 mm | Intermediate Z height used during the rise phase of a pick, to prevent the object from being dragged sideways. |
| `ArmController.MOVE_DELAY` | 1.2 s | Wait time after sending each non-transit motion packet, allowing the arm to reach the commanded position before the next command is sent. |

### 11.3 Arduino Firmware Parameters

| Parameter | Default | Description |
|---|---|---|
| `STEP_SIZE` | `1` degree | Maximum angular change per servo per firmware tick. Increasing this speeds up motion but reduces smoothness. |
| `INTERVAL` | `10` ms | Firmware tick period. Lower values increase update frequency; higher values reduce CPU load on the Arduino. |
| Serial baud rate | `115200` | Must exactly match `ArmController.BAUD_RATE` in `finalv1.py`. |
| Gripper closed angle | `85` degrees | Target servo angle when the gripper is commanded to close. Adjust to tune grip force. |
| Gripper open angle | `145` degrees | Target servo angle when the gripper is commanded to open. |

---

## 12. Design Decisions and Engineering Notes

**Why HSV colour space instead of RGB for object detection?**

RGB colour values are a product of both surface colour and illumination intensity. A red object under dim lighting produces RGB values that can be indistinguishable from a dark-coloured object under bright lighting. HSV separates chromatic identity (hue) from brightness (value) and colourfulness (saturation), making it possible to define colour ranges that reliably identify a hue across a wide range of lighting intensities using a single calibrated threshold set. This is a well-established principle in industrial machine vision systems and was chosen deliberately over RGB thresholding or neural-network-based segmentation for its interpretability, tunability, and low computational overhead.

**Why a locally-running LLM instead of a rule-based command parser?**

A rule-based parser would require explicit enumeration of every grammatical construction a user might employ. Users vary significantly in how they phrase instructions: "pick up the red one", "grab the red circle", "move the red thing on the left", and "take the red circle to the right zone" all express the same operation but share little surface-level syntactic similarity. A language model handles this linguistic variation as an inherent capability, without requiring the programmer to anticipate every phrasing. Running the model locally via Ollama ensures the system is fully offline and not subject to external API latency, rate limits, or service outages. Setting `temperature = 0` ensures the planner produces the same output for the same input, which is essential for reproducible debugging.

**Why multi-frame temporal fusion instead of single-frame detection?**

A single camera frame is susceptible to transient noise that no post-processing can distinguish from a true object. Camera vibration during the capture instant, a brief specular reflection from overhead lighting, or a shadow from the user's hand can cause a genuine object to be missed or a spurious detection to be produced. Capturing five frames over a 0.6-second window and applying majority-vote fusion over that time window robustly suppresses single-frame transients while adding only modest capture latency. The 5-frame default was chosen as the minimum number that allows a 3-of-5 majority vote to override any single outlier frame.

**Why firmware-side rate limiting rather than host-side interpolated delays?**

Controlling servo speed from the Python host (e.g. sending incremental angle targets with `time.sleep()` calls between them) would block the Python process during motion, preventing it from handling other tasks. It would also be susceptible to scheduling jitter from the host OS, which does not provide real-time guarantees. The Arduino firmware's independent tick-based rate limiter runs on a dedicated microcontroller with deterministic timing, decoupling motion control from the host entirely. The Python code can send a target angle at any time; the firmware handles smooth motion to that target automatically.

**Why closed-form analytical IK instead of numerical methods?**

Analytical (closed-form) inverse kinematics computes exact joint angles in constant time using trigonometry. It is fully deterministic, produces no iteration artifacts, and allows joint limit violations to be detected analytically before any motor command is sent. Numerical IK methods (such as Jacobian pseudo-inverse or gradient descent) require iterative convergence, can converge to local minima or fail to converge entirely near singularities, and have variable and unpredictable computation times. For a 4-degree-of-freedom arm with a known geometric structure, the analytical solution is strictly superior.

---

## 13. Known Limitations and Future Work

**Lighting sensitivity**

Despite HSV-based detection, the colour thresholds were tuned for a specific indoor daylight environment. Strong coloured artificial light sources, direct sunlight, or fluorescent lighting with a strong colour cast can shift the perceived hue of objects and require re-tuning the `COLOR_RANGES` dictionary. A future improvement would be to add an automatic white-balance correction step before HSV conversion, or to implement an adaptive threshold calibration routine.

**Object overlap and occlusion**

The vision pipeline processes contours as independent entities and has no occlusion model. Two adjacent or overlapping objects whose colour masks merge will be detected as a single large contour and may be classified as a different shape or rejected altogether. A future improvement would be to apply a watershed or instance segmentation algorithm to separate touching objects.

**LLM output reliability**

The language model occasionally produces malformed JSON or references non-existent object IDs, particularly for complex multi-step or ambiguous commands. The executor handles these cases by logging the error and aborting the current plan, but it does not retry automatically. A future improvement would add a retry loop with a modified prompt that explicitly reports the previous parse failure to the model.

**Fixed workspace height**

The system assumes all objects rest on a flat surface at a known, constant Z height (`Z_GRIP = -75 mm`). It cannot handle objects of different heights, objects on elevated platforms, or stacked configurations. Integrating a depth camera (e.g. Intel RealSense) would allow per-object height measurement and dynamic Z targeting.

**No object identity persistence between commands**

The vision pipeline runs a complete fresh detection on every command. Object identifiers (`obj_0`, `obj_1`, ...) are reassigned at each detection pass based on the order objects are found. Objects that are moved, added, or removed between commands are correctly handled, but a reference to `obj_0` in one command does not necessarily refer to the same physical object as `obj_0` in the next. A future improvement would maintain a persistent object registry that tracks objects across commands using spatial proximity matching.

**Single arm configuration**

The IK solver always uses the elbow-up configuration (`elbow_up = True`). Some workspace positions are reachable only in the elbow-down configuration. Adding automatic configuration selection based on the target position would extend the effective workspace.

---

## 14. Troubleshooting

**"No Arduino port found" on startup**

The system could not detect the Arduino's USB serial port automatically. Check that:
- The USB cable is fully seated at both ends.
- The Arduino has been successfully flashed with `IK1.ino` (if the upload failed silently, the firmware may not be running).
- The correct USB driver is installed for the Arduino's USB-to-serial chip (CH340 drivers are commonly required for Arduino Uno clones on Windows).

To bypass auto-detection and specify the port directly, modify `finalv1.py` to instantiate `ArmController(port='COM3')` on Windows or `ArmController(port='/dev/ttyUSB0')` on Linux.

**"calibration.npz not found — XY will be in pixels"**

The homography calibration file is missing. Run `a4_calibration.py`, complete the four-corner selection, and press S to save the file. Ensure the file is saved in the same directory as `finalv1.py`.

**Arm moves to wrong positions**

The arm consistently places objects at incorrect locations. Possible causes and remedies:
- The camera resolution has changed since calibration. Re-run `a4_calibration.py` at the current resolution.
- The A4 sheet was moved after calibration. The sheet only needs to be present during calibration; it can be removed during normal operation, but its position at calibration time defines the coordinate origin. If the arm's position relative to the sheet has changed, recalibrate.
- The `SERVO_CAL` constants do not match the physical arm. Verify the servo zero positions by sending individual test packets.

**Vision detects no objects despite objects being present**

- Confirm the camera is open to the correct device index (`CAMERA_SRC`). If another application is using the camera, `cv2.VideoCapture` may open silently but return black frames.
- Check that the workspace lighting falls within the range for which `COLOR_RANGES` was tuned. Use `a4_calibration.py`'s live view to inspect the raw camera feed.
- Confirm the detected objects are within the workspace bounds (`WORKSPACE_X`, `WORKSPACE_Y`).

**LLM produces invalid JSON**

The model did not follow the output format instruction. Try:
- Switching to a model with stronger instruction-following (e.g. `llama3:8b` or `mistral`).
- Simplifying the command phrasing.
- Increasing `num_ctx` in the `call_llm_planner()` options if the context is being truncated.

**Servos jitter continuously**

The servos are not receiving stable power. Ensure the servo power supply is separate from the Arduino's 5V rail and can supply sufficient current. Six MG996R servos under load can draw over 3 A combined.

---

## 15. Repository Structure

```
VoiceArm/
|
|-- IK1.ino                  Arduino firmware: serial parsing, step-rate-limited servo control
|-- a4_calibration.py        Interactive camera calibration tool: homography computation and saving
|-- finalv1.py               Main controller: vision, STT, LLM planning, IK, arm execution
|
|-- calibration.npz          Saved homography matrix (generated at runtime; not committed to VCS)
|-- command.wav              Temporary voice recording file (generated at runtime; not committed)
|-- snapshot_debug.jpg       Last annotated detection frame, saved after each scene scan
|
|-- README.md                This document
|-- LICENSE                  Project licence terms
```

Files marked "generated at runtime" are produced during operation and should be excluded from version control by adding them to `.gitignore`:

```
calibration.npz
command.wav
snapshot_debug.jpg
__pycache__/
*.pyc
voicearm-env/
```

---

## 16. Acknowledgements

### 3D Mechanical Model

The 3D-printed arm structure used in this project is based on the open-source model published by **doganhobby**:

> [https://github.com/doganhobby/Arduino-Projects/tree/main/3D%20Robot%20Arm](https://github.com/doganhobby/Arduino-Projects/tree/main/3D%20Robot%20Arm)

Only the mechanical model files (STL/CAD) were used from this repository. No code, firmware, or software from that project is present in VoiceArm. All control software — the Arduino firmware, the calibration tool, the vision pipeline, the IK solver, the LLM interface, and the execution loop — was designed and implemented independently by the VoiceArm team.

### Third-Party Software

| Library / Tool | Version | Purpose | Source |
|---|---|---|---|
| OpenAI Whisper | latest | Offline speech recognition | https://github.com/openai/whisper |
| Ollama | latest | Local LLM inference runtime | https://ollama.com |
| Gemma 4 (`gemma4:e4b`) | e4b quantisation | Natural language task planning | Google DeepMind, distributed via Ollama |
| OpenCV (`opencv-python`) | 4.x | Computer vision, image processing, homography | https://opencv.org |
| NumPy | 1.x / 2.x | Numerical array operations, matrix computation | https://numpy.org |
| SciPy (`scipy.io.wavfile`) | 1.x | WAV file writing for audio capture | https://scipy.org |
| PySerial | 3.x | USB serial communication with Arduino | https://pyserial.readthedocs.io |
| sounddevice | 0.4.x | Real-time audio capture from microphone | https://python-sounddevice.readthedocs.io |

---

## 17. License

This project is released for academic and educational use. See `LICENSE` for full terms.
