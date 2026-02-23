

# 👁️🖐️ Vision OS V4: Hand-Tracking Gesture Engine

**Vision OS V4** is an advanced Python-based computer vision engine that turns your standard webcam into a spatial computing interface. It maps physical hand gestures to operating system commands using MediaPipe and OpenCV.

Unlike standard trackers that jump and stutter when you click, Vision OS V4 features a **Dynamically Auto-Switching Hybrid Engine**. It instantly detects whether you are using one hand or two, applying specialized kinematic smoothing, twitch absorption, and coordinate freezing to guarantee pixel-perfect, jitter-free cursor control.

---

## ✨ V4 Core Features

* **Auto-Switching Hybrid Modes:** Seamlessly transitions between 1-Hand Mode (casual browsing) and 2-Hand Pro Mode (high-precision dual-hand control).
* **Twitch Absorption (1-Hand):** A 0.15-second coordinate freeze during pinches completely eliminates cursor micro-jitters when clicking.
* **Absolute Click-Freeze (2-Hand):** The tracking hand totally locks cursor coordinates while the secondary hand executes clicks, ensuring absolute precision.
* **Global Sleep Mode:** A dedicated standby toggle that bypasses heavy math and rendering loops, freeing up your CPU for other tasks while you watch movies or chill.
* **Active Tracking Box:** Uses `numpy.interp` to map a defined ergonomic box in your webcam feed directly to your monitor, requiring minimal physical arm movement.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/chaz-ux/vision-os-gesture-engine.git
cd vision-os-gesture-engine

```


2. **Install the required dependencies:**
```bash
pip install opencv-python mediapipe pyautogui screen-brightness-control numpy

```


3. **Run the engine:**
```bash
python vision_os.py

```



*(Note: Ensure your terminal/IDE has permission to control your mouse/keyboard and access your camera).*

---

## 🎛️ The Gesture Manual

### 🖱️ Modes & Navigation

| Action | Gesture | Description |
| --- | --- | --- |
| **System Sleep / Wake** | 👎 **Thumbs Down** | Fold fingers into a fist, point thumb straight down. Pauses the tracking engine to save CPU. |
| **1-Hand Mode** | ☝️ **One Hand on Camera** | AI routes all movement and clicking logic to your active hand. |
| **2-Hand Mode (Pro)** | 👐 **Two Hands on Camera** | AI routes cursor movement to your Right hand, and clicking logic to your Left hand. |
| **Move Cursor** | ☝️ **Point** | Point with just your **Index Finger**. |
| **Scroll** | ✌️ **Two Fingers** | Hold your **Index and Middle** fingers up (Thumb tucked). Move hand up/down. |

### 🎯 Clicking Mechanics

| Action | Gesture | Description |
| --- | --- | --- |
| **Left Click / Drag** | 🤏 **Index Pinch** | Pinch your **Index Finger** and **Thumb** together. *(In 2-Hand Mode, do this with your Left Hand).* |
| **Right Click** | 🤌 **Middle Pinch** | Pinch your **Middle Finger** and **Thumb** together underneath your index finger. |

### 🔊 System & Media Controls

| Action | Gesture | Description |
| --- | --- | --- |
| **Volume Control** | 🤙 **The "Shaka"** | Stick your **Thumb and Pinky** out. Twist your wrist up to increase volume, down to decrease. |
| **Brightness Control** | ✌️ **Scout Sign** | Hold **Index, Middle, and Ring** fingers up. Tilt hand up to brighten, down to dim. |
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

By calculating the Euclidean distance between these specific nodes (e.g., node `4` for the thumb tip and node `8` for the index tip), the engine determines if a finger is extended or if a specific pinch configuration is met.

To achieve smooth cursor movement, the script uses a `collections.deque` to maintain a rolling average of your hand's X/Y coordinates across 5 frames, filtering out optical noise before sending commands to `pyautogui`.

---

## ⚠️ Troubleshooting

* **Cursor is jumping wildly:** Ensure your room is well-lit. Shadows can confuse the depth sensors.
* **Gestures aren't triggering:** Face your palm directly toward the camera. The model is trained primarily on palm-forward and slight-angle data.
* **Brightness control throws an error:** Ensure you are running on a supported OS for `screen-brightness-control` (Windows/Linux primary displays).

