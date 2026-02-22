import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from collections import deque

# --- BRIGHTNESS CONTROL ---
try:
    import screen_brightness_control as sbc
    HAS_SBC = True
except ImportError:
    HAS_SBC = False
    print("WARNING: 'screen_brightness_control' not found. Brightness gestures disabled.")

# --- VISION OS V4 HYBRID SETTINGS ---
FRAME_R = 120                # Active Tracking Box margin (Pixels from edge)
SMOOTHING_FRAMES = 5         # Rolling Average window for jitter-free movement
PINCH_RATIO = 0.6            
SWIPE_THRESH = 70            
# ------------------------------

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

# Shared State Tracking
history_x = deque(maxlen=SMOOTHING_FRAMES)
history_y = deque(maxlen=SMOOTHING_FRAMES)
wrist_history_x = deque(maxlen=5)
wrist_history_y = deque(maxlen=5)

cursor_x, cursor_y = screen_w / 2, screen_h / 2 
prev_ix, prev_iy = 0, 0
dragging = False

# V2 Two-Hand State
is_frozen = False
frozen_x, frozen_y = 0, 0

# V3 One-Hand State
pinch_start_time = 0 

cooldowns = {
    'play': 0, 'swipe': 0, 'vol': 0, 'bright': 0, 'sys_keys': 0, 'right_click': 0
}

def get_dist(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

print("VISION OS V4 HYBRID Online. Auto-Switching Active (1-Hand & 2-Hand Modes).")

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # Draw the Active Tracking Box
    cv2.rectangle(img, (FRAME_R, FRAME_R), (w - FRAME_R, h - FRAME_R), (255, 0, 255), 2)
    cv2.putText(img, "ACTIVE TRACKING AREA", (FRAME_R, FRAME_R - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Global Failsafe: Drop clicks if hands vanish
    if not results.multi_hand_landmarks and dragging:
        pyautogui.mouseUp()
        dragging = False

    left_hand = None
    right_hand = None
    mode = 0 # 0 = Idle, 1 = One-Handed, 2 = Two-Handed
    
    # Click Triggers
    left_click_held = False
    right_click_triggered = False

    if results.multi_hand_landmarks:
        # Sort hands by X coordinate to strictly separate Left and Right
        hands_sorted = sorted(results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)
        
        if len(hands_sorted) == 2:
            left_hand = hands_sorted[0]  
            right_hand = hands_sorted[1] 
            mode = 2
            cv2.putText(img, "MODE: 2-HAND (PRO)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(hands_sorted) == 1:
            right_hand = hands_sorted[0] 
            mode = 1
            cv2.putText(img, "MODE: 1-HAND", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ==========================================
        # 1. PROCESS LEFT HAND (Only in 2-Hand Mode)
        # ==========================================
        if mode == 2 and left_hand:
            mp_draw.draw_landmarks(img, left_hand, mp_hands.HAND_CONNECTIONS)
            lm = left_hand.landmark
            palm_width = get_dist(lm[5], lm[17], w, h)
            
            idx_pinch = (get_dist(lm[4], lm[8], w, h) / palm_width) < PINCH_RATIO
            mid_pinch = (get_dist(lm[4], lm[12], w, h) / palm_width) < PINCH_RATIO
            
            if idx_pinch: left_click_held = True
            if mid_pinch and not idx_pinch: right_click_triggered = True

        # ==========================================
        # 2. PROCESS ACTIVE HAND (Right/Single Hand)
        # ==========================================
        if right_hand:
            mp_draw.draw_landmarks(img, right_hand, mp_hands.HAND_CONNECTIONS)
            lm = right_hand.landmark
            
            # SENSOR ARRAY
            fingers = [
                1 if get_dist(lm[0], lm[8], w, h) > get_dist(lm[0], lm[6], w, h) * 1.15 else 0,
                1 if get_dist(lm[0], lm[12], w, h) > get_dist(lm[0], lm[10], w, h) * 1.15 else 0,
                1 if get_dist(lm[0], lm[16], w, h) > get_dist(lm[0], lm[14], w, h) * 1.15 else 0,
                1 if get_dist(lm[0], lm[20], w, h) > get_dist(lm[0], lm[18], w, h) * 1.15 else 0
            ]
            thumb_out = 1 if lm[4].x < lm[3].x else 0 
            
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            wx, wy = int(lm[0].x * w), int(lm[0].y * h)
            wrist_history_x.append(wx)
            wrist_history_y.append(wy)

            # --- STATE 0: IDLE / CLUTCH ---
            if sum(fingers) == 0 and not thumb_out:
                cv2.putText(img, "CLUTCH (FROZEN)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                prev_ix, prev_iy = 0, 0 
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                continue

            # --- ESCAPE KEY ---
            if fingers == [1, 1, 0, 0] and thumb_out:
                cv2.putText(img, "ESCAPE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                if time.time() - cooldowns['sys_keys'] > 1.5:
                    pyautogui.press('esc')
                    cooldowns['sys_keys'] = time.time()
                continue

            # --- WINDOWS KEY ---
            if fingers == [0, 1, 1, 1]:
                cv2.putText(img, "START MENU", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                if time.time() - cooldowns['sys_keys'] > 1.5:
                    pyautogui.press('win')
                    cooldowns['sys_keys'] = time.time()
                continue

            # --- STATE 1: VOLUME KNOB ---
            if fingers == [0, 0, 0, 1] and thumb_out:
                cv2.putText(img, "AUDIO CONTROL", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if time.time() - cooldowns['vol'] > 0.1: 
                    if lm[4].y < lm[17].y: 
                        pyautogui.press('volumeup')
                        cv2.circle(img, (int(lm[4].x*w), int(lm[4].y*h)), 15, (0, 255, 0), cv2.FILLED)
                    else: 
                        pyautogui.press('volumedown')
                        cv2.circle(img, (int(lm[4].x*w), int(lm[4].y*h)), 15, (0, 0, 255), cv2.FILLED)
                    cooldowns['vol'] = time.time()
                continue
                
            # --- STATE 2: BRIGHTNESS KNOB ---
            if fingers == [1, 1, 1, 0] and HAS_SBC:
                cv2.putText(img, "BRIGHTNESS", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if time.time() - cooldowns['bright'] > 0.1:
                    current_brightness = sbc.get_brightness(display=0)[0]
                    if lm[8].y < lm[0].y - 0.2: 
                        sbc.set_brightness(min(100, current_brightness + 5))
                        cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                    elif lm[8].y > lm[0].y - 0.1: 
                        sbc.set_brightness(max(0, current_brightness - 5))
                        cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
                    cooldowns['bright'] = time.time()
                continue

            # --- STATE 3: MEDIA PLAY/PAUSE ---
            if fingers == [1, 0, 0, 1] and thumb_out:
                cv2.putText(img, "MEDIA OVERRIDE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                if time.time() - cooldowns['play'] > 1.5: 
                    pyautogui.press('playpause')
                    cooldowns['play'] = time.time()
                continue

            # --- STATE 4: COMMAND SWIPES ---
            if sum(fingers) == 4 and thumb_out:
                cv2.putText(img, "COMMAND MODE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
                if len(wrist_history_x) == 5 and (time.time() - cooldowns['swipe'] > 1.0):
                    dx = wrist_history_x[4] - wrist_history_x[0]
                    dy = wrist_history_y[4] - wrist_history_y[0]
                    if dx > SWIPE_THRESH: pyautogui.hotkey('win', 'right'); cooldowns['swipe'] = time.time()
                    elif dx < -SWIPE_THRESH: pyautogui.hotkey('win', 'left'); cooldowns['swipe'] = time.time()
                    elif dy > SWIPE_THRESH: pyautogui.hotkey('win', 'd'); cooldowns['swipe'] = time.time()
                continue

            # --- STATE 5: SCROLLING ---
            if fingers == [1, 1, 0, 0] and not thumb_out:
                cv2.putText(img, "SCROLLING", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if prev_iy == 0: prev_iy = iy
                dy = iy - prev_iy
                if abs(dy) > 2: pyautogui.scroll(-int(dy * 5))
                prev_iy = iy 
                prev_ix = 0 
                if dragging: pyautogui.mouseUp(); dragging = False
                continue

            # --- STATE 6: NAVIGATION & DYNAMIC CLICKS ---
            if fingers == [1, 0, 0, 0] or (mode == 1 and dragging):
                prev_ix, prev_iy = ix, iy # Reset scroll history
                
                # Interpolate from Active Box to Full Monitor
                mapped_x = np.interp(ix, (FRAME_R, w - FRAME_R), (0, screen_w))
                mapped_y = np.interp(iy, (FRAME_R, h - FRAME_R), (0, screen_h))
                
                # >> MODE 2: ABSOLUTE CLICK FREEZE <<
                if mode == 2:
                    if left_click_held or right_click_triggered:
                        if not is_frozen:
                            frozen_x, frozen_y = cursor_x, cursor_y
                            is_frozen = True
                        cursor_x, cursor_y = frozen_x, frozen_y
                        cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
                    else:
                        is_frozen = False
                        history_x.append(mapped_x)
                        history_y.append(mapped_y)
                        cursor_x = sum(history_x) / len(history_x)
                        cursor_y = sum(history_y) / len(history_y)
                        cv2.circle(img, (ix, iy), 10, (255, 255, 0), cv2.FILLED)

                # >> MODE 1: TWITCH ABSORPTION <<
                elif mode == 1:
                    palm_w = get_dist(lm[5], lm[17], w, h)
                    idx_pinch = (get_dist(lm[4], lm[8], w, h) / palm_w) < PINCH_RATIO
                    mid_pinch = (get_dist(lm[4], lm[12], w, h) / palm_w) < PINCH_RATIO
                    
                    if idx_pinch:
                        left_click_held = True
                        if not dragging:
                            pinch_start_time = time.time()
                        
                        # Absorb twitch for 0.15s
                        if time.time() - pinch_start_time < 0.15 and len(history_x) > 0:
                            mapped_x = history_x[-1]
                            mapped_y = history_y[-1]
                        cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                    else:
                        cv2.circle(img, (ix, iy), 10, (255, 255, 0), cv2.FILLED)
                    
                    if mid_pinch and not idx_pinch:
                        right_click_triggered = True

                    history_x.append(mapped_x)
                    history_y.append(mapped_y)
                    cursor_x = sum(history_x) / len(history_x)
                    cursor_y = sum(history_y) / len(history_y)

                # Move Mouse
                pyautogui.moveTo(cursor_x, cursor_y, _pause=False)

    # ==========================================
    # 3. GLOBAL CLICK EXECUTION
    # ==========================================
    if left_click_held:
        click_text = "LEFT CLICK (LEFT HAND)" if mode == 2 else "LEFT CLICK (1-HAND)"
        cv2.putText(img, click_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if not dragging:
            pyautogui.mouseDown()
            dragging = True
    else:
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    if right_click_triggered and (time.time() - cooldowns['right_click'] > 0.5):
        cv2.putText(img, "RIGHT CLICK", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        pyautogui.rightClick()
        cooldowns['right_click'] = time.time()

    cv2.imshow("VISION OS V4 HYBRID", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()