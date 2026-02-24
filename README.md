👁️🖐️ Vision OS V5 Pro: Advanced Kinematic Gesture Engine

Vision OS V5 Pro is an advanced Python-based computer vision engine that turns your standard webcam into a spatial computing interface. It maps physical hand gestures to operating system commands using MediaPipe, OpenCV, and advanced kinematic smoothing.

Unlike standard trackers that jump and stutter, V5 introduces a Dynamic EMA (Exponential Moving Average) Smoother with a built-in physical deadzone, an Intentionality Engine to prevent accidental clicks, and Edge-Clamped Navigation allowing effortless access to your entire screen.

✨ V5 Core Features

Edge-to-Edge Navigation (NEW): Utilizes numpy.interp clamping on a defined visual boundary. Reaching the edge of the purple on-screen box maps perfectly to the edge of your monitor, allowing easy access to the taskbar and screen corners without your hand leaving the camera frame.

Intentionality Engine (NEW): Heavy commands (like window switching via Open Palm) now require a brief "Hold to Engage" timer. This completely eliminates accidental window-snapping when simply stretching or opening your hand.

Micro-Jitter Deadzone (NEW): The cursor mathematically ignores pixel-level wrist twitches. It remains frozen until deliberate gliding movement is detected, providing a "heavy," ultra-precise feel.

Velocity-Based Swipes (NEW): Swiping to change tracks or switch desktops now calculates actual hand velocity over a history buffer, rather than just raw distance.

Hysteresis Anti-Flicker: Finger states require a multi-frame buffer to register a change, stopping the AI from rapidly flickering between gestures if it loses confidence.

Auto-Switching Hybrid Modes: Seamlessly transitions between 1-Hand Mode (casual browsing) and 2-Hand Pro Mode (high-precision dual-hand control).

🛠️ Installation & Setup

Clone the repository:

git clone [https://github.com/chaz-ux/vision-os-gesture-engine.git](https://github.com/chaz-ux/vision-os-gesture-engine.git)
cd vision-os-gesture-engine


Install the required dependencies:

pip install opencv-python mediapipe pyautogui screen-brightness-control numpy


Run the engine:

python vision_os_pro.py


(Note: Ensure your terminal/IDE has permission to control your mouse/keyboard and access your camera).

🎛️ The Gesture Manual

🖱️ Modes & Navigation

Action

Gesture

Description

System Sleep / Wake

👎 Thumbs Down

Fold fingers into a fist, point thumb straight down. Pauses the tracking engine to save CPU.

1-Hand Mode

☝️ One Hand on Camera

AI routes all movement and clicking logic to your active hand.

2-Hand Mode (Pro)

👐 Two Hands on Camera

AI routes cursor movement to your Right hand, and clicking logic to your Left hand.

Move Cursor

☝️ Point

Point with just your Index Finger. Reach the edges of the purple tracking box to reach the edges of your screen.

Auto-Scroll

✌️ Two Fingers

Hold Index and Middle fingers up (Thumb tucked). Move hand above or below the center "deadzone" to kinetically scroll up/down.

🎯 Clicking Mechanics

Action

Gesture

Description

Left Click / Drag

🤏 Index Pinch

Pinch your Index Finger and Thumb together. (In 2-Hand Mode, do this with your Left Hand).

Right Click

🤌 Middle Pinch

Pinch your Middle Finger and Thumb together underneath your index finger.

🔊 System & Media Controls

Action

Gesture

Description

Volume Control

🤙 The "Shaka"

Stick your Thumb and Pinky out. Twist your wrist up to increase volume, down to decrease.

Brightness Control

✌️ Scout Sign

Hold Index, Middle, and Ring fingers up. Tilt hand up to brighten, down to dim.

Play / Pause

🕸️ "Web-Shooter"

Hold Index and Pinky fingers up with Thumb out. Swipe left/right rapidly to change tracks.

🖥️ Window Management & Hotkeys

Action

Gesture

Description

Command Mode (Snap/Desk)

✋ Stop Sign (Hold)

Open hand completely and hold for 0.25s. Once engaged, swipe fast Left/Right to snap windows, or Down for desktop.

Escape Key

✌️ + 👍 Peace + Thumb

Hold Index and Middle fingers up, with Thumb sticking out.

Start Menu (Win)

👌 "OK" Sign

Pinch Index and Thumb together, leaving Middle, Ring, and Pinky fingers up.

⚙️ Under the Hood

Vision OS V5 Pro relies on MediaPipe's hand landmark detection, tracking 21 distinct 3D landmarks on your hand in real-time.

The State Machine (HandAnalyzer):
Instead of trusting raw frame-by-frame data, V5 routes finger detection through a Hysteresis buffer. A finger must consistently read as "Open" or "Closed" for 5 consecutive frames before the system registers a state change.

Kinematics (DynamicSmoother):
Raw X/Y coordinates are passed through an Exponential Moving Average (EMA) algorithm. The alpha (smoothing weight) dynamically scales based on how fast your hand is moving. Move fast, and smoothing drops to zero for zero-latency tracking. Move slow, and smoothing increases drastically to allow surgical precision. A strict physical deadzone prevents micro-jitters from registering at all.

⚠️ Troubleshooting

Cannot reach the edges of the screen: Ensure your hand is moving all the way to the edges of the purple Active Tracking Boundary drawn on the camera feed. You can adjust Config.Tracking.FRAME_MARGIN in the code to make this box smaller/easier to reach.

Cursor feels stuck / heavy: This is the Deadzone filter preventing jitter. If it requires too much force to move the mouse, lower Config.Cursor.DEADZONE in the script.

Gestures aren't triggering: Face your palm directly toward the camera. Ensure your room is well-lit.

Brightness control throws an error: Ensure you are running on a supported OS for screen-brightness-control (Windows/Linux primary displays).