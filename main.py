import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from collections import deque

# --- NEW REQUIREMENT ---
# Run in terminal: pip install screen-brightness-control
try:
    import screen_brightness_control as sbc
    HAS_SBC = True
except ImportError:
    HAS_SBC = False
    print("WARNING: 'screen_brightness_control' not found. Brightness gestures will be disabled.")

# --- THE VISION OS SETTINGS ---
BASE_SENSITIVITY = 1.2       
ACCELERATION = 0.3           
SMOOTHING = 0.3              
PINCH_RATIO = 0.6            
SWIPE_THRESH = 70            
# ------------------------------

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

# State Tracking
cursor_x, cursor_y = screen_w / 2, screen_h / 2 
target_x, target_y = cursor_x, cursor_y         
prev_ix, prev_iy = 0, 0
dragging = False
wrist_history_x = deque(maxlen=5)
wrist_history_y = deque(maxlen=5)

# Cooldowns to prevent gesture spamming
cooldowns = {
    'play': 0, 'swipe': 0, 'vol': 0, 'bright': 0, 'sys_keys': 0, 'right_click': 0
}

def get_dist(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

print("VISION RC-1.1 Online. Ready for GitHub Commit.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm = hand_lms.landmark
            
            # --- 1. SENSOR ARRAY (Strict 1.15x extension threshold) ---
            fingers = [
                1 if get_dist(lm[0], lm[8], w, h) > get_dist(lm[0], lm[6], w, h) * 1.15 else 0,   # Index
                1 if get_dist(lm[0], lm[12], w, h) > get_dist(lm[0], lm[10], w, h) * 1.15 else 0, # Middle
                1 if get_dist(lm[0], lm[16], w, h) > get_dist(lm[0], lm[14], w, h) * 1.15 else 0, # Ring
                1 if get_dist(lm[0], lm[20], w, h) > get_dist(lm[0], lm[18], w, h) * 1.15 else 0  # Pinky
            ]
            thumb_out = 1 if lm[4].x < lm[3].x else 0 # (Flipped for right hand)
            
            palm_width = get_dist(lm[5], lm[17], w, h)
            
            # Core Pinches
            idx_pinch = (get_dist(lm[4], lm[8], w, h) / palm_width) < PINCH_RATIO
            mid_pinch = (get_dist(lm[4], lm[12], w, h) / palm_width) < PINCH_RATIO

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            wx, wy = int(lm[0].x * w), int(lm[0].y * h)
            wrist_history_x.append(wx)
            wrist_history_y.append(wy)

            # --- 2. GESTURE ENGINE ---

            # STATE 0: IDLE / CLUTCH (Fist)
            if sum(fingers) == 0 and not thumb_out:
                cv2.putText(img, "CLUTCH (FROZEN)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                prev_ix, prev_iy = 0, 0 
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                continue

            # --- NEW SYSTEM KEYS ---
            # ESCAPE KEY ("Peace Out" Sign - Index & Middle Up, Thumb Out)
            if fingers == [1, 1, 0, 0] and thumb_out:
                cv2.putText(img, "ESCAPE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                if time.time() - cooldowns['sys_keys'] > 1.5:
                    pyautogui.press('esc')
                    cooldowns['sys_keys'] = time.time()
                prev_ix, prev_iy = 0, 0
                continue

            # WINDOWS KEY ("OK" Sign - Middle, Ring, Pinky Up. Index pinched to Thumb)
            if fingers == [0, 1, 1, 1]:
                cv2.putText(img, "START MENU", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                if time.time() - cooldowns['sys_keys'] > 1.5:
                    pyautogui.press('win')
                    cooldowns['sys_keys'] = time.time()
                prev_ix, prev_iy = 0, 0
                continue

            # STATE 1: VOLUME KNOB (Shaka: Strictly Thumb + Pinky)
            if fingers == [0, 0, 0, 1] and thumb_out:
                cv2.putText(img, "AUDIO CONTROL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if time.time() - cooldowns['vol'] > 0.1: 
                    if lm[4].y < lm[17].y: 
                        pyautogui.press('volumeup')
                        cv2.circle(img, (int(lm[4].x*w), int(lm[4].y*h)), 15, (0, 255, 0), cv2.FILLED)
                    else: 
                        pyautogui.press('volumedown')
                        cv2.circle(img, (int(lm[4].x*w), int(lm[4].y*h)), 15, (0, 0, 255), cv2.FILLED)
                    cooldowns['vol'] = time.time()
                prev_ix, prev_iy = 0, 0
                continue
                
            # STATE 2: BRIGHTNESS KNOB (Index, Middle, Ring up)
            if fingers == [1, 1, 1, 0] and HAS_SBC:
                cv2.putText(img, "BRIGHTNESS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if time.time() - cooldowns['bright'] > 0.1:
                    current_brightness = sbc.get_brightness(display=0)[0]
                    if lm[8].y < lm[0].y - 0.2: 
                        sbc.set_brightness(min(100, current_brightness + 5))
                        cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                    elif lm[8].y > lm[0].y - 0.1: 
                        sbc.set_brightness(max(0, current_brightness - 5))
                        cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
                    cooldowns['bright'] = time.time()
                prev_ix, prev_iy = 0, 0
                continue

            # STATE 3: MEDIA PLAY/PAUSE (Spider-Man: Strictly Index + Pinky)
            if fingers == [1, 0, 0, 1] and thumb_out:
                cv2.putText(img, "MEDIA OVERRIDE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                if time.time() - cooldowns['play'] > 1.5: 
                    pyautogui.press('playpause')
                    cooldowns['play'] = time.time()
                prev_ix, prev_iy = 0, 0
                continue

           # STATE 4: COMMAND SWIPES (All 5 fingers up)
            if sum(fingers) == 4 and thumb_out:
                cv2.putText(img, "COMMAND MODE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
                if len(wrist_history_x) == 5 and (time.time() - cooldowns['swipe'] > 1.0):
                    dx = wrist_history_x[4] - wrist_history_x[0]
                    dy = wrist_history_y[4] - wrist_history_y[0]
                    
                    if dx > SWIPE_THRESH: # Swipe Right -> Snap Window Right
                        pyautogui.hotkey('win', 'right')
                        cooldowns['swipe'] = time.time()
                    elif dx < -SWIPE_THRESH: # Swipe Left -> Snap Window Left
                        pyautogui.hotkey('win', 'left')
                        cooldowns['swipe'] = time.time()
                    elif dy > SWIPE_THRESH: # Swipe Down -> Show Desktop
                        pyautogui.hotkey('win', 'd') 
                        cooldowns['swipe'] = time.time()
                prev_ix, prev_iy = 0, 0
                continue
            
            # STATE 5: SCROLLING (Index + Middle Up, Thumb Tucked)
            if fingers == [1, 1, 0, 0] and not thumb_out:
                cv2.putText(img, "SCROLLING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if prev_iy == 0: prev_iy = iy
                dy = iy - prev_iy
                if abs(dy) > 2: 
                    pyautogui.scroll(-int(dy * 5))
                prev_iy = iy 
                prev_ix = 0 
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                continue

            # STATE 6: NAVIGATION & CLICKS (Only Index Up)
            if fingers == [1, 0, 0, 0]:
                if prev_ix == 0: 
                    prev_ix, prev_iy = ix, iy 
                
                # Kinematic Movement
                dx, dy = ix - prev_ix, iy - prev_iy
                speed = math.hypot(dx, dy)
                dynamic_sens = BASE_SENSITIVITY + (speed * ACCELERATION)
                
                target_x += dx * dynamic_sens
                target_y += dy * dynamic_sens
                target_x = max(0, min(screen_w, target_x))
                target_y = max(0, min(screen_h, target_y))
                
                # Low-Pass Filter Smoothing 
                cursor_x += (target_x - cursor_x) * SMOOTHING
                cursor_y += (target_y - cursor_y) * SMOOTHING
                
                pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
                prev_ix, prev_iy = ix, iy

                # Left Click / Drag (Index + Thumb)
                if idx_pinch:
                    cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                else:
                    cv2.circle(img, (ix, iy), 10, (255, 255, 0), 2)
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                # Right Click (Middle + Thumb underneath)
                if mid_pinch and not idx_pinch and (time.time() - cooldowns['right_click'] > 0.5):
                    cv2.circle(img, int(lm[12].x*w), int(lm[12].y*h), 20, (0, 0, 255), cv2.FILLED)
                    pyautogui.rightClick()
                    cooldowns['right_click'] = time.time()

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("VISION OS", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()