# Vision OS: Advanced Hand-Tracking Gesture Engine 👁️🖐️

Vision OS is a Python-based relative tracking engine that turns your webcam into a spatial interface. It maps physical hand gestures to kinematic mouse movements and system commands using MediaPipe and OpenCV. 

Unlike standard absolute trackers, Vision uses a depth-agnostic "relative" tracking system (anchor points) combined with a low-pass smoothing filter, allowing for butter-smooth, jitter-free cursor control.

## 🛠️ Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe pyautogui screen-brightness-control

3. Run the script: python vision_os.py (or whatever your filename is).
🎛️ The Gesture Manual
Navigation & Clicks
Move Cursor: Point with just your Index Finger.

Left Click / Drag: Pinch your Index Finger and Thumb together.

Right Click: Rub your Middle Finger and Thumb together underneath your index finger.

Scroll: Hold your Index and Middle fingers up (Thumb tucked). Move hand up/down.

System & Media Controls
Volume Control (The "Shaka"): Stick your Thumb and Pinky out. Twist your wrist up to increase volume, down to decrease.

Brightness Control (The "Scout"): Hold Index, Middle, and Ring fingers up. Tilt hand up to brighten, down to dim.

Play / Pause (The "Web-Shooter"): Hold Index and Pinky fingers up with Thumb out.

Window Management & Hotkeys
Window Snapping: Open hand completely (Stop sign). Swipe Left or Right to snap windows to the sides of your screen. Swipe Down to minimize everything and show the desktop.

Escape Key (The "Peace Sign"): Hold Index and Middle fingers up, with Thumb sticking out.

Windows / Start Menu (The "OK Sign"): Pinch Index and Thumb together, leaving Middle, Ring, and Pinky fingers up.   