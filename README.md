
<div align="center">

# 👁️🖐️ Vision OS V5 Pro

**Advanced Kinematic Gesture Engine for Spatial Computing**

*Transform your standard webcam into a high-precision spatial computing interface. Map physical hand gestures to operating system commands using MediaPipe, OpenCV, and advanced kinematic smoothing.*

[Features](https://www.google.com/search?q=%23-v5-pro-core-features) • [Installation](https://www.google.com/search?q=%23%25EF%25B8%258F-installation--setup) • [Gesture Manual](https://www.google.com/search?q=%23-the-gesture-matrix) • [Under the Hood](https://www.google.com/search?q=%23-under-the-hood) • [Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)

</div>

## 🚀 The Vision OS Difference

Unlike standard optical trackers that jump, stutter, and accidentally click when you move your hand, **Vision OS V5 Pro** introduces robust mathematical filtering. By utilizing a Dynamic Exponential Moving Average (EMA) smoother, built-in physical deadzones, and an Intentionality Engine, V5 provides a heavy, ultra-precise, "surgical" feel previously reserved for dedicated spatial hardware.

## ✨ V5 Pro Core Features

* **Edge-Clamped Spatial Navigation:** Utilizes `numpy.interp` clamping on a defined visual boundary. Reaching the edge of the on-screen tracking box maps perfectly to the edge of your monitor—access your taskbar and screen corners without your hand ever leaving the camera frame.
* **Intentionality Engine:** Heavy commands (like window snapping via Open Palm) require a brief "Hold to Engage" timer (0.25s). This completely eliminates accidental UI triggers when simply stretching or opening your hand.
* **Kinematic Micro-Jitter Deadzone:** The cursor mathematically ignores pixel-level wrist twitches. It remains frozen until deliberate, gliding movement is detected.
* **Velocity-Based Swipes:** Swiping to change tracks or switch desktops calculates actual hand velocity over a dynamic history buffer, rather than relying on raw distance.
* **Hysteresis Anti-Flicker:** Finger states require a multi-frame buffer to register a change, stopping the AI from rapidly flickering between states if camera confidence drops.
* **Auto-Switching Hybrid Modes:** Seamlessly transitions between **1-Hand Mode** (casual browsing) and **2-Hand Pro Mode** (high-precision dual-hand control) on the fly.

## 🛠️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/chaz-ux/vision-os-gesture-engine.git
cd vision-os-gesture-engine

```

**2. Install required dependencies**

```bash
pip install opencv-python mediapipe pyautogui screen-brightness-control numpy

```

**3. Run the engine**

```bash
python vision_os_pro.py

```

> **Note:** Ensure your terminal/IDE has the necessary OS permissions to control your mouse/keyboard and access your webcam.

## 🎛️ The Gesture Matrix

### 🖱️ Modes & Navigation

| Action | Gesture | Description |
| --- | --- | --- |
| **System Sleep / Wake** | 👎 **Thumbs Down** | Fold fingers into a fist, point thumb straight down. Pauses the tracking engine to save CPU. |
| **1-Hand Mode** | ☝️ **One Hand Active** | AI routes all movement and clicking logic to your active hand. |
| **2-Hand Mode (Pro)** | 👐 **Two Hands Active** | Routes cursor movement to **Right Hand**, clicking logic to **Left Hand**. |
| **Move Cursor** | ☝️ **Point** | Point with Index Finger. Reach the edges of the tracking box to reach screen edges. |
| **Auto-Scroll** | ✌️ **Two Fingers** | Hold Index & Middle fingers up. Move hand above/below the center deadzone to kinetically scroll. |

### 🎯 Clicking Mechanics

| Action | Gesture | Description |
| --- | --- | --- |
| **Left Click / Drag** | 🤏 **Index Pinch** | Pinch Index Finger and Thumb together. *(Do this with Left Hand in 2-Hand Mode).* |
| **Right Click** | 🤌 **Middle Pinch** | Pinch Middle Finger and Thumb together underneath the index finger. |

### 🔊 System & Media Controls

| Action | Gesture | Description |
| --- | --- | --- |
| **Volume Control** | 🤙 **The "Shaka"** | Thumb and Pinky out. Twist your wrist up to increase volume, down to decrease. |
| **Brightness** | ✌️ **Scout Sign** | Index, Middle, and Ring fingers up. Tilt hand up to brighten, down to dim. |
| **Play / Pause** | 🕸️ **Web-Shooter** | Index and Pinky fingers up, Thumb out. Swipe left/right rapidly to change tracks. |

### 🖥️ Window Management & Hotkeys

| Action | Gesture | Description |
| --- | --- | --- |
| **Command Mode** | ✋ **Stop Sign (Hold)** | Open hand completely and hold for `0.25s`. Once engaged, swipe Left/Right to snap windows, or Down for desktop. |
| **Escape Key** | ✌️+👍 **Peace + Thumb** | Hold Index and Middle fingers up, with Thumb sticking out. |
| **Start Menu (Win)** | 👌 **"OK" Sign** | Pinch Index and Thumb together, leaving Middle, Ring, and Pinky fingers up. |

## 🧠 Under the Hood

Vision OS V5 Pro relies on MediaPipe's hand landmark detection, tracking 21 distinct 3D landmarks in real-time. We bridge the gap between raw data and smooth UX using two core architectures:

1. **The State Machine (`HandAnalyzer`)**
Instead of trusting raw frame-by-frame data, V5 routes finger detection through a **Hysteresis buffer**. A finger must consistently read as "Open" or "Closed" for 5 consecutive frames before the system registers a state change.
2. **Advanced Kinematics (`DynamicSmoother`)**
Raw X/Y coordinates pass through an Exponential Moving Average (EMA) algorithm. The smoothing weight () dynamically scales based on your hand's velocity. Move fast, and smoothing drops to zero for latency-free tracking. Move slow, and smoothing increases drastically to allow for surgical precision.

## ⚠️ Troubleshooting & Configuration

| Issue | Solution |
| --- | --- |
| **Cannot reach the edges of the screen** | Ensure your hand reaches the edges of the purple *Active Tracking Boundary* on the camera feed. Adjust `Config.Tracking.FRAME_MARGIN` in the code to make this box smaller/easier to reach. |
| **Cursor feels stuck or heavy** | This is the Deadzone filter preventing jitter. If it requires too much physical force to move the mouse, lower `Config.Cursor.DEADZONE` in the script. |
| **Gestures aren't triggering** | Face your palm directly toward the camera. Ensure your room is well-lit and your background isn't overly cluttered. |
| **Brightness control throws an error** | Ensure you are running on a supported OS for `screen-brightness-control` (Windows/Linux primary displays only). |

<div align="center">
<i>Engineered for the future of spatial computing.</i>
</div>