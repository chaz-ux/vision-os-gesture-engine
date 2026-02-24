import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import socket
from collections import deque
import ctypes

# --- OS DPI FIX ---
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# --- BRIGHTNESS CONTROL ---
try:
    import screen_brightness_control as sbc
    HAS_SBC = True
except ImportError:
    HAS_SBC = False
    print("WARNING: 'screen_brightness_control' not found.")

# --- SYSTEM CONFIGURATION ---
pyautogui.FAILSAFE = False

# ---> UDP Socket Setup (Talks to C#)
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_mouse_cmd(cmd_string):
    """Fires a packet to C# for instant mouse execution"""
    try:
        sock.sendto(cmd_string.encode('utf-8'), (UDP_IP, UDP_PORT))
    except:
        pass

class Config:
    class Tracking:
        CUSTOM_SCREEN_W = None 
        CUSTOM_SCREEN_H = None 
        
        AREA_TOP = 0.25      
        AREA_BOTTOM = 0.80   
        AREA_LEFT = 0.30     
        AREA_RIGHT = 0.85    
        STATE_BUFFER_SIZE = 5 
        
    class Cursor:
        PINCH_ENGAGE_RATIO = 0.18    
        PINCH_RELEASE_RATIO = 0.35   
        CLICK_FREEZE_TIME = 0.2     
        DEADZONE = 1.5   
        CLICK_DROP_DELAY = 0.15           
        
    class Gestures:
        INTENT_DELAY = 0.25         
        SWIPE_VELOCITY_THRESH = 90  
        SCROLL_MAX_SPEED = 150      
        COOLDOWN_DEFAULT = 1.0      
        COOLDOWN_FAST = 0.1         

class MathUtils:
    @staticmethod
    def get_dist(p1, p2, w=1, h=1):
        return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

class Smoother1D:
    def __init__(self, alpha=0.2):
        self.value = 0.0
        self.alpha = alpha
        
    def update(self, target):
        self.value = self.value + self.alpha * (target - self.value)
        return self.value

class DynamicSmoother:
    def __init__(self, min_alpha=0.2, max_alpha=0.95, speed_scale=80.0):
        self.prev_x = None
        self.prev_y = None
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.speed_scale = speed_scale

    def update(self, curr_x, curr_y):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = curr_x, curr_y
            return curr_x, curr_y

        dist = math.hypot(curr_x - self.prev_x, curr_y - self.prev_y)
        
        if dist < Config.Cursor.DEADZONE:
            return self.prev_x, self.prev_y

        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * min(1.0, dist / self.speed_scale)
        smoothed_x = self.prev_x + alpha * (curr_x - self.prev_x)
        smoothed_y = self.prev_y + alpha * (curr_y - self.prev_y)
        
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y

class HandAnalyzer:
    def __init__(self):
        self.finger_history = [deque(maxlen=Config.Tracking.STATE_BUFFER_SIZE) for _ in range(4)]
        self.thumb_history = deque(maxlen=Config.Tracking.STATE_BUFFER_SIZE)
        self.current_fingers = [0, 0, 0, 0]
        self.current_thumb = 0

    def analyze(self, hand, w, h):
        lm = hand.landmark
        palm_h = MathUtils.get_dist(lm[0], lm[9], w, h)
        palm_w = MathUtils.get_dist(lm[5], lm[17], w, h)
        
        raw_fingers = [
            1 if MathUtils.get_dist(lm[0], lm[8], w, h) > palm_h * 1.5 else 0,
            1 if MathUtils.get_dist(lm[0], lm[12], w, h) > palm_h * 1.5 else 0,
            1 if MathUtils.get_dist(lm[0], lm[16], w, h) > palm_h * 1.4 else 0,
            1 if MathUtils.get_dist(lm[0], lm[20], w, h) > palm_h * 1.3 else 0 
        ]
        
        thumb_pinky_dist = MathUtils.get_dist(lm[4], lm[17], w, h)
        raw_thumb = 1 if thumb_pinky_dist > (palm_w * 1.6) else 0

        for i in range(4):
            self.finger_history[i].append(raw_fingers[i])
            if sum(self.finger_history[i]) == Config.Tracking.STATE_BUFFER_SIZE:
                self.current_fingers[i] = 1
            elif sum(self.finger_history[i]) == 0:
                self.current_fingers[i] = 0

        self.thumb_history.append(raw_thumb)
        if sum(self.thumb_history) == Config.Tracking.STATE_BUFFER_SIZE:
            self.current_thumb = 1
        elif sum(self.thumb_history) == 0:
            self.current_thumb = 0

        return self.current_fingers, self.current_thumb, lm

class HUDManager:
    @staticmethod
    def draw_tracking_box(img, x1, y1, x2, y2):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, "ACTIVE TRACKING BOUNDARY", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)

    @staticmethod
    def draw_status(img, text, color=(255, 255, 255), pos=(20, 70), scale=1.0):
        cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, 2)

    @staticmethod
    def draw_cursor(img, x, y, color=(255, 255, 0), radius=10):
        cv2.circle(img, (x, y), radius, color, cv2.FILLED)
        cv2.circle(img, (x, y), radius+3, (0, 0, 0), 2)

class GestureController:
    def __init__(self):
        self.cooldowns = {
            'play': 0, 'window_swipe': 0, 'vol': 0, 'bright': 0, 
            'sys_keys': 0, 'right_click': 0, 'sleep': 0, 'media_swipe': 0
        }
        self.wrist_history_x = deque(maxlen=10)
        self.wrist_history_y = deque(maxlen=10)
        
        self.is_sleeping = False
        self.palm_open_start_time = 0
        self.command_mode_active = False
        self.scroll_smoother = Smoother1D(alpha=0.15)
        
        self.is_clutched_now = False 

    def process_system_gestures(self, img, fingers, thumb_out, lm, w, h, is_dragging):
        current_time = time.time()
        wx, wy = int(lm[0].x * w), int(lm[0].y * h)
        self.is_clutched_now = False
        
        self.wrist_history_x.append(wx)
        self.wrist_history_y.append(wy)

        if sum(fingers) == 0 and lm[4].y > lm[0].y + 0.1:
            if current_time - self.cooldowns['sleep'] > Config.Gestures.COOLDOWN_DEFAULT:
                self.is_sleeping = not self.is_sleeping
                self.cooldowns['sleep'] = current_time
                if is_dragging: 
                    send_mouse_cmd("LUP")
                    is_dragging = False # FIX 1: Properly reset drag state when putting system to sleep
            HUDManager.draw_status(img, "TOGGLING SLEEP MODE...", (0, 165, 255))
            return True, is_dragging

        if self.is_sleeping:
            HUDManager.draw_status(img, "SYSTEM ASLEEP : THUMBS DOWN TO WAKE", (0, 0, 255), scale=0.8)
            return True, is_dragging

        if sum(fingers) == 0 and not thumb_out:
            HUDManager.draw_status(img, "CLUTCH (LIFT MOUSE)", (0, 0, 255))
            self.is_clutched_now = True
            if is_dragging:
                send_mouse_cmd("LUP")
                is_dragging = False
            self.scroll_smoother.update(0) 
            self.command_mode_active = False
            return True, is_dragging

        if sum(fingers) == 4 and thumb_out:
            if self.palm_open_start_time == 0:
                self.palm_open_start_time = current_time
            
            if current_time - self.palm_open_start_time > Config.Gestures.INTENT_DELAY:
                self.command_mode_active = True
                HUDManager.draw_status(img, "COMMAND MODE ACTIVE", (255, 100, 100))
                
                if len(self.wrist_history_x) == self.wrist_history_x.maxlen:
                    dx = self.wrist_history_x[-1] - self.wrist_history_x[0]
                    dy = self.wrist_history_y[-1] - self.wrist_history_y[0]
                    
                    if current_time - self.cooldowns['window_swipe'] > Config.Gestures.COOLDOWN_DEFAULT:
                        if abs(dx) > abs(dy): 
                            if dx > Config.Gestures.SWIPE_VELOCITY_THRESH:
                                pyautogui.hotkey('win', 'right')
                                self.cooldowns['window_swipe'] = current_time
                            elif dx < -Config.Gestures.SWIPE_VELOCITY_THRESH:
                                pyautogui.hotkey('win', 'left')
                                self.cooldowns['window_swipe'] = current_time
                        else: 
                            if dy > Config.Gestures.SWIPE_VELOCITY_THRESH:
                                pyautogui.hotkey('win', 'd')
                                self.cooldowns['window_swipe'] = current_time
                            elif dy < -Config.Gestures.SWIPE_VELOCITY_THRESH:
                                pyautogui.hotkey('win', 'up')
                                self.cooldowns['window_swipe'] = current_time
            else:
                HUDManager.draw_status(img, "HOLD TO ENGAGE...", (200, 200, 200))
            return True, is_dragging
        else:
            self.palm_open_start_time = 0
            self.command_mode_active = False

        if fingers == [1, 1, 0, 0] and thumb_out:
            HUDManager.draw_status(img, "ESCAPE", (255, 255, 255))
            if current_time - self.cooldowns['sys_keys'] > Config.Gestures.COOLDOWN_DEFAULT:
                pyautogui.press('esc')
                self.cooldowns['sys_keys'] = current_time
            return True, is_dragging

        if fingers == [0, 1, 1, 1]:
            HUDManager.draw_status(img, "START MENU", (255, 255, 255))
            if current_time - self.cooldowns['sys_keys'] > Config.Gestures.COOLDOWN_DEFAULT:
                pyautogui.press('win')
                self.cooldowns['sys_keys'] = current_time
            return True, is_dragging

        if fingers == [0, 0, 0, 1] and thumb_out:
            HUDManager.draw_status(img, "AUDIO CONTROL", (0, 255, 255))
            if current_time - self.cooldowns['vol'] > Config.Gestures.COOLDOWN_FAST:
                if lm[4].y < lm[20].y: pyautogui.press('volumeup')
                else: pyautogui.press('volumedown')
                self.cooldowns['vol'] = current_time
            return True, is_dragging

        if fingers == [1, 1, 1, 0] and HAS_SBC:
            HUDManager.draw_status(img, "BRIGHTNESS", (255, 255, 255))
            if current_time - self.cooldowns['bright'] > Config.Gestures.COOLDOWN_FAST:
                current_brightness = sbc.get_brightness(display=0)[0]
                if lm[8].y < lm[0].y - 0.2: sbc.set_brightness(min(100, current_brightness + 5))
                elif lm[8].y > lm[0].y - 0.1: sbc.set_brightness(max(0, current_brightness - 5))
                self.cooldowns['bright'] = current_time
            return True, is_dragging

        if fingers == [1, 0, 0, 1] and thumb_out:
            HUDManager.draw_status(img, "MEDIA CONTROL", (255, 0, 255))
            if len(self.wrist_history_x) == self.wrist_history_x.maxlen:
                dx = self.wrist_history_x[-1] - self.wrist_history_x[0]
                if dx > Config.Gestures.SWIPE_VELOCITY_THRESH and (current_time - self.cooldowns['media_swipe'] > 1.0):
                    pyautogui.press('nexttrack')
                    self.cooldowns['media_swipe'] = current_time
                    self.cooldowns['play'] = current_time + 1.0 
                elif dx < -Config.Gestures.SWIPE_VELOCITY_THRESH and (current_time - self.cooldowns['media_swipe'] > 1.0):
                    pyautogui.press('prevtrack')
                    self.cooldowns['media_swipe'] = current_time
                    self.cooldowns['play'] = current_time + 1.0
                # FIX 2: Added a much longer 2-second cooldown to Play/Pause to prevent spamming while holding the sign
                elif abs(dx) < 30 and (current_time - self.cooldowns['play'] > 2.0): 
                    pyautogui.press('playpause')
                    self.cooldowns['play'] = current_time
            return True, is_dragging

        if fingers == [1, 1, 0, 0] and not thumb_out:
            HUDManager.draw_status(img, "AUTO SCROLL", (255, 255, 0))
            hand_y = lm[9].y
            deadzone_top = 0.40
            deadzone_bottom = 0.60
            target_speed = 0
            
            if hand_y < deadzone_top:
                intensity = (deadzone_top - hand_y) / deadzone_top
                target_speed = int(intensity * Config.Gestures.SCROLL_MAX_SPEED)
            elif hand_y > deadzone_bottom:
                intensity = (hand_y - deadzone_bottom) / (1.0 - deadzone_bottom)
                target_speed = -int(intensity * Config.Gestures.SCROLL_MAX_SPEED)

            actual_speed = int(self.scroll_smoother.update(target_speed))
            
            if abs(actual_speed) > 0:
                send_mouse_cmd(f"SCROLL|{actual_speed * 2}")
                
            center_h = int(h / 2)
            scroll_offset = int((actual_speed / Config.Gestures.SCROLL_MAX_SPEED) * 100)
            cv2.line(img, (50, center_h), (50, center_h - scroll_offset), (255, 255, 0), 10)
            
            if is_dragging: 
                send_mouse_cmd("LUP")
                is_dragging = False
            return True, is_dragging

        self.scroll_smoother.update(0)
        return False, is_dragging


class VisionOS:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.screen_w, self.screen_h = pyautogui.size()
        if Config.Tracking.CUSTOM_SCREEN_W is not None:
            self.screen_w = Config.Tracking.CUSTOM_SCREEN_W
        if Config.Tracking.CUSTOM_SCREEN_H is not None:
            self.screen_h = Config.Tracking.CUSTOM_SCREEN_H
        
        self.smoother = DynamicSmoother()
        self.analyzer = HandAnalyzer()
        self.gestures = GestureController()
        
        self.dragging = False
        self.frozen_x, self.frozen_y = self.screen_w / 2, self.screen_h / 2
        self.is_frozen = False
        self.pinch_start_time = 0
        self.last_pinch_time = 0
        
        self.mouse_offset_x = 0
        self.mouse_offset_y = 0
        self.was_clutched = False
        self.last_cursor_x = self.screen_w / 2
        self.last_cursor_y = self.screen_h / 2

    def run(self):
        print(f"VISION OS HYBRID Online. Mapping to {self.screen_w}x{self.screen_h} display.")
        
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success: break
            
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            box_l = int(w * Config.Tracking.AREA_LEFT)
            box_r = int(w * Config.Tracking.AREA_RIGHT)
            box_t = int(h * Config.Tracking.AREA_TOP)
            box_b = int(h * Config.Tracking.AREA_BOTTOM)
            
            HUDManager.draw_tracking_box(img, box_l, box_t, box_r, box_b)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = self.hands.process(img_rgb)
            img_rgb.flags.writeable = True
            
            if not results.multi_hand_landmarks and self.dragging:
                send_mouse_cmd("LUP")
                self.dragging = False

            mode = 0
            left_hand = right_hand = None
            left_click_held = right_click_triggered = False

            if results.multi_hand_landmarks:
                hands_sorted = sorted(results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)
                if len(hands_sorted) == 2:
                    left_hand, right_hand = hands_sorted[0], hands_sorted[1]
                    mode = 2
                    HUDManager.draw_status(img, "SYSTEM MODE: 2-HAND (PRO)", (0, 255, 0), (20, 30), 0.7)
                else:
                    right_hand = hands_sorted[0] 
                    mode = 1
                    HUDManager.draw_status(img, "SYSTEM MODE: 1-HAND", (255, 255, 0), (20, 30), 0.7)

                if mode == 2 and left_hand and not self.gestures.is_sleeping:
                    self.mp_draw.draw_landmarks(img, left_hand, self.mp_hands.HAND_CONNECTIONS)
                    lm = left_hand.landmark
                    # FIX 3: Added max(..., 1e-6) to prevent fatal ZeroDivisionError when hand is perfectly sideways
                    palm_w = max(MathUtils.get_dist(lm[5], lm[17], w, h), 1e-6) 
                    # ---> MODE 2 PINCH LOGIC IS HERE <---
                    idx_dist_ratio = MathUtils.get_dist(lm[4], lm[8], w, h) / palm_w     # Index-to-Thumb
                    mid_dist_ratio = MathUtils.get_dist(lm[4], lm[12], w, h) / palm_w    # Middle-to-Thumb
        
                    # Left Click (Index + Thumb)
                    if self.dragging:
                     if idx_dist_ratio < Config.Cursor.PINCH_RELEASE_RATIO: left_click_held = True
                    else:
                     if idx_dist_ratio < Config.Cursor.PINCH_ENGAGE_RATIO: left_click_held = True
            
                     # Right Click (Middle + Thumb)
                     if mid_dist_ratio < Config.Cursor.PINCH_ENGAGE_RATIO and not left_click_held: 
                      right_click_triggered = True

                if right_hand:
                    self.mp_draw.draw_landmarks(img, right_hand, self.mp_hands.HAND_CONNECTIONS)
                    fingers, thumb_out, lm = self.analyzer.analyze(right_hand, w, h)
                    
                    gesture_handled, self.dragging = self.gestures.process_system_gestures(
                        img, fingers, thumb_out, lm, w, h, self.dragging
                    )

                    if self.gestures.is_clutched_now:
                        self.was_clutched = True

                    if not gesture_handled and not self.gestures.is_sleeping:
                        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                        
                        mapped_x = np.interp(ix, [box_l, box_r], [0, self.screen_w])
                        mapped_y = np.interp(iy, [box_t, box_b], [0, self.screen_h])

                        if self.was_clutched:
                            self.mouse_offset_x = self.last_cursor_x - mapped_x
                            self.mouse_offset_y = self.last_cursor_y - mapped_y
                            self.was_clutched = False
                            
                            self.smoother.prev_x = self.last_cursor_x
                            self.smoother.prev_y = self.last_cursor_y

                        target_x = mapped_x + self.mouse_offset_x
                        target_y = mapped_y + self.mouse_offset_y

                        target_x = np.clip(target_x, 0, self.screen_w)
                        target_y = np.clip(target_y, 0, self.screen_h)

                        if mode == 2:
                            if left_click_held or right_click_triggered:
                                if not self.is_frozen:
                                    # FIX 4: Protects against 'NoneType' crash if a user pinches before the smoother registers a frame
                                    self.frozen_x = self.smoother.prev_x if self.smoother.prev_x is not None else self.last_cursor_x
                                    self.frozen_y = self.smoother.prev_y if self.smoother.prev_y is not None else self.last_cursor_y
                                    self.is_frozen = True
                                cursor_x, cursor_y = self.frozen_x, self.frozen_y
                                HUDManager.draw_cursor(img, ix, iy, color=(0, 0, 255), radius=15)
                            else:
                                self.is_frozen = False
                                cursor_x, cursor_y = self.smoother.update(target_x, target_y)
                                HUDManager.draw_cursor(img, ix, iy, color=(255, 255, 0), radius=10)

                        elif mode == 1:
                        # ---> MODE 1 PINCH LOGIC IS HERE <---
                            palm_w = max(MathUtils.get_dist(lm[5], lm[17], w, h), 1e-6)
                            idx_dist_ratio = MathUtils.get_dist(lm[4], lm[8], w, h) / palm_w     # Index-to-Thumb
                            mid_dist_ratio = MathUtils.get_dist(lm[4], lm[12], w, h) / palm_w    # Middle-to-Thumb
            
                            # Left Click (Index + Thumb)
                            if self.dragging:
                              if idx_dist_ratio < Config.Cursor.PINCH_RELEASE_RATIO: left_click_held = True
                            else:
                              if idx_dist_ratio < Config.Cursor.PINCH_ENGAGE_RATIO: left_click_held = True
            
                            # ... (cursor freeze logic is here) ...
            
                            # Right Click (Middle + Thumb)
                            if mid_dist_ratio < Config.Cursor.PINCH_ENGAGE_RATIO and not left_click_held: 
                              right_click_triggered = True
                                
                            cursor_x, cursor_y = self.smoother.update(target_x, target_y)

                        self.last_cursor_x = cursor_x
                        self.last_cursor_y = cursor_y

                        send_mouse_cmd(f"MOVE|{int(cursor_x)}|{int(cursor_y)}")

            if not self.gestures.is_sleeping:
                if left_click_held:
                    HUDManager.draw_status(img, f"LEFT CLICK ({'LEFT HAND' if mode == 2 else '1-HAND'})", (0, 255, 0), (20, 110), 0.8)
                    if not self.dragging:
                        send_mouse_cmd("LDOWN")
                        self.dragging = True
                else:
                    if self.dragging:
                        send_mouse_cmd("LUP") 
                        self.dragging = False

                if right_click_triggered and (time.time() - self.gestures.cooldowns['right_click'] > 0.5):
                    HUDManager.draw_status(img, "RIGHT CLICK", (0, 0, 255), (20, 150), 0.8)
                    send_mouse_cmd("RCLICK") 
                    self.gestures.cooldowns['right_click'] = time.time()

            cv2.imshow("VISION OS HYBRID ENGINE", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VisionOS()
    app.run()