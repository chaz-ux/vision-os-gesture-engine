
# 👁️🖐️ Vision OS: Advanced Hand-Tracking Gesture Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://google.github.io/mediapipe/)

**Vision OS** is a Python-based relative tracking engine that turns your standard webcam into a spatial, minority-report-style interface. It maps physical hand gestures to kinematic mouse movements and system commands using MediaPipe and OpenCV. 

Unlike standard absolute trackers that jump and stutter, Vision uses a depth-agnostic "relative" tracking system (anchor points) combined with a low-pass smoothing filter. The result? **Butter-smooth, jitter-free cursor control.**

---

## ✨ Core Features
* **Kinematic Smoothing:** Eliminates micro-jitters for precise clicking.
* **Dynamic Sensitivity:** Cursor accelerates based on your hand speed.
* **No Special Hardware:** Runs purely on CPU using any standard 720p/1080p webcam.
* **Zero-Touch Interface:** Control media, virtual desktops, window snapping, and brightness without touching your keyboard.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/chaz-ux/vision-os-gesture-engine.git](https://github.com/chaz-ux/vision-os-gesture-engine.git)
   cd vision-os-gesture-engine

```

2. **Install the required dependencies:**
```bash
pip install opencv-python mediapipe pyautogui screen-brightness-control

```


3. **Run the engine:**
```bash
python vision_os.py

```


*(Note: Ensure your terminal/IDE has permission to control your mouse/keyboard and access your camera).*

---

## 🎛️ The Gesture Manual

### 🖱️ Navigation & Clicks

| Action | Gesture | Description |
| --- | --- | --- |
| **Move Cursor** | ☝️ **Point** | Point with just your **Index Finger**. |
| **Left Click / Drag** | 🤏 **Index Pinch** | Pinch your **Index Finger** and **Thumb** together. |
| **Right Click** | 🤌 **Middle Pinch** | Pinch your **Middle Finger** and **Thumb** together underneath your index finger. |
| **Scroll** | ✌️ **Two Fingers** | Hold your **Index and Middle** fingers up (Thumb tucked). Move hand up/down. |

### 🔊 System & Media Controls

| Action | Gesture | Description |
| --- | --- | --- |
| **Volume Control** | 🤙 **The "Shaka"** | Stick your **Thumb and Pinky** out. Twist your wrist up to increase volume, down to decrease. |
| **Brightness Control** | Scout Sign | Hold **Index, Middle, and Ring** fingers up. Tilt hand up to brighten, down to dim. |
| **Play / Pause** | 🕸️ **"Web-Shooter"** | Hold **Index and Pinky** fingers up with Thumb out. |

### 🖥️ Window Management & Hotkeys

| Action | Gesture | Description |
| --- | --- | --- |
| **Window Snapping** | ✋ **Stop Sign** | Open hand completely. Swipe **Left or Right** to snap windows. Swipe **Down** to show the desktop. |
| **Escape Key** | ✌️ + 👍 **Peace + Thumb** | Hold **Index and Middle** fingers up, with Thumb sticking out. |
| **Start Menu (Win)** | 👌 **"OK" Sign** | Pinch **Index and Thumb** together, leaving Middle, Ring, and Pinky fingers up. |

---

## ⚙️ Under the Hood

Vision OS relies on MediaPipe's hand landmark detection. It tracks 21 distinct 3D landmarks on your hand in real-time.

By calculating the Euclidean distance between these specific nodes (e.g., node `4` for the thumb tip and node `8` for the index tip), the engine determines if a finger is extended or if a specific pinch configuration is met, ensuring high accuracy even in varying light conditions.



## ⚠️ Troubleshooting

* **Cursor is jumping wildly:** Ensure your room is well-lit. Shadows can confuse the depth sensors.
* **Gestures aren't triggering:** Face your palm directly toward the camera. The model is trained primarily on palm-forward and slight-angle data.
* **Brightness control throws an error:** Ensure you are running on a supported OS for `screen-brightness-control` (Windows/Linux primary displays).



