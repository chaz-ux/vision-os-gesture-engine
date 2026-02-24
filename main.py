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

# --- SYSTEM CONFIGURATION ---
pyautogui.FAILSAFE = False  # Handled safely by numpy bounds clamping
pyautogui.PAUSE = 0         # Crucial for ZERO lag cursor movement

class Config:
    """Categorized settings for easy tweaking."""
    class Tracking:
        # Bounding box margins (Pixels from camera edge). 
        # Increase these if you want a smaller physical area to cover the whole screen.
        FRAME_MARGIN_X = 120  
        FRAME_MARGIN_Y = 100  
        STATE_BUFFER_SIZE = 5 # Frames required to confirm a finger state (Anti-Flicker)
        
    class Cursor:
        PINCH_RATIO = 0.30          # STRICTER: 30% of palm width to trigger click.
        CLICK_FREEZE_TIME = 0.2     # Seconds to freeze cursor on click to absorb twitch
        DEADZONE = 1.5              # Minimum pixel movement to register as intentional (Anti-Jitter)
        
    class Gestures:
        INTENT_DELAY = 0.25         # Seconds to hold Open Palm before Command Mode activates
        SWIPE_VELOCITY_THRESH = 90  # Distance wrist must travel over history buffer for a swipe
        SCROLL_MAX_SPEED = 150      # Maximum scroll ticks per cycle
        COOLDOWN_DEFAULT = 1.0      # Default timeout between heavy actions
        COOLDOWN_FAST = 0.1         # Timeout for scroll/vol/bright ticks

class MathUtils:
    @staticmethod
    def get_dist(p1, p2, w=1, h=1):
        """Calculates Euclidean distance between two MediaPipe landmarks."""
        return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

class Smoother1D:
    """1D Exponential Moving Average for values like Scroll Speed or Volume."""
    def __init__(self, alpha=0.2):
        self.value = 0.0
        self.alpha = alpha
        
    def update(self, target):
        self.value = self.value + self.alpha * (target - self.value)
        return self.value

class DynamicSmoother:
    """
    Dynamic Exponential Moving Average (EMA) for 2D Cursor Movement. 
    Includes a Deadzone to filter out unintentional micro-twitches.
    """
    def __init__(self, min_alpha=0.03, max_alpha=0.8, speed_scale=80.0):
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
        
        # Intentionality Filter: If the movement is tiny, ignore it to stop jitter
        if dist < Config.Cursor.DEADZONE:
            return self.prev_x, self.prev_y

        # Scale alpha based on distance moved (faster moves = less smoothing = lower latency)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * min(1.0, dist / self.speed_scale)
        
        smoothed_x = self.prev_x + alpha * (curr_x - self.prev_x)
        smoothed_y = self.prev_y + alpha * (curr_y - self.prev_y)
        
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y

class HandAnalyzer:
    """
    Advanced hand state tracking with hysteresis buffers to prevent AI flickering.
    """
    def __init__(self):
        self.finger_history = [deque(maxlen=Config.Tracking.STATE_BUFFER_SIZE) for _ in range(4)]
        self.thumb_history = deque(maxlen=Config.Tracking.STATE_BUFFER_SIZE)
        self.current_fingers = [0, 0, 0, 0]
        self.current_thumb = 0

    def analyze(self, hand, w, h):
        """Converts raw landmarks into stable, debounced finger states."""
        lm = hand.landmark
        
        # Base distances for relative measuring (adapts to hand distance from camera)
        palm_h = MathUtils.get_dist(lm[0], lm[9], w, h)
        palm_w = MathUtils.get_dist(lm[5], lm[17], w, h)
        
        # Raw binary reads (Is the fingertip further from the wrist than the palm base?)
        raw_fingers = [
            1 if MathUtils.get_dist(lm[0], lm[8], w, h) > palm_h * 1.3 else 0,  # Index
            1 if MathUtils.get_dist(lm[0], lm[12], w, h) > palm_h * 1.3 else 0, # Middle
            1 if MathUtils.get_dist(lm[0], lm[16], w, h) > palm_h * 1.25 else 0, # Ring
            1 if MathUtils.get_dist(lm[0], lm[20], w, h) > palm_h * 1.15 else 0  # Pinky
        ]
        
        # Robust Thumb Read: Distance from pinky knuckle to thumb tip compared to palm width
        thumb_pinky_dist = MathUtils.get_dist(lm[4], lm[17], w, h)
        raw_thumb = 1 if thumb_pinky_dist > (palm_w * 1.5) else 0

        # Apply Hysteresis (Anti-flicker filter)
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
    """Handles all drawing and visual feedback on the camera feed."""
    @staticmethod
    def draw_tracking_box(img, w, h):
        # Draw the active mapping area. Reaching these lines hits the edge of your monitor.
        cv2.rectangle(img, (Config.Tracking.FRAME_MARGIN_X, Config.Tracking.FRAME_MARGIN_Y), 
                      (w - Config.Tracking.FRAME_MARGIN_X, h - Config.Tracking.FRAME_MARGIN_Y), 
                      (255, 0, 255), 2)
        cv2.putText(img, "ACTIVE TRACKING BOUNDARY", 
                    (Config.Tracking.FRAME_MARGIN_X, Config.Tracking.FRAME_MARGIN_Y - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)

    @staticmethod
    def draw_status(img, text, color=(255, 255, 255), pos=(20, 70), scale=1.0):
        # Adds a slight drop shadow for readability against bright backgrounds
        cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, 2)

    @staticmethod
    def draw_cursor(img, x, y, color=(255, 255, 0), radius=10):
        cv2.circle(img, (x, y), radius, color, cv2.FILLED)
        cv2.circle(img, (x, y), radius+3, (0, 0, 0), 2) # Outline

class GestureController:
    """
    Centralized hub for processing distinct hand formations into OS-level commands.
    """
    def __init__(self):
        self.cooldowns = {
            'play': 0, 'window_swipe': 0, 'vol': 0, 'bright': 0, 
            'sys_keys': 0, 'right_click': 0, 'sleep': 0, 'media_swipe': 0
        }
        self.wrist_history_x = deque(maxlen=10) # Smaller buffer = faster velocity read
        self.wrist_history_y = deque(maxlen=10)
        
        self.is_sleeping = False
        self.palm_open_start_time = 0 # Tracks intentionality of command mode
        self.command_mode_active = False
        
        self.scroll_smoother = Smoother1D(alpha=0.15)

    def process_system_gestures(self, img, fingers, thumb_out, lm, w, h, is_dragging):
        """
        The core state machine. Evaluates stabilized finger states and executes OS commands.
        Returns (gesture_handled_bool, dragging_state_bool).
        """
        current_time = time.time()
        wx, wy = int(lm[0].x * w), int(lm[0].y * h)
        
        # Track wrist position for velocity-based swipe calculations
        self.wrist_history_x.append(wx)
        self.wrist_history_y.append(wy)

        # --- 1. SLEEP TOGGLE (THUMBS DOWN) ---
        if sum(fingers) == 0 and lm[4].y > lm[0].y + 0.1: # Thumb distinctly below wrist
            if current_time - self.cooldowns['sleep'] > Config.Gestures.COOLDOWN_DEFAULT:
                self.is_sleeping = not self.is_sleeping
                self.cooldowns['sleep'] = current_time
                if is_dragging: pyautogui.mouseUp()
            HUDManager.draw_status(img, "TOGGLING SLEEP MODE...", (0, 165, 255))
            return True, is_dragging

        if self.is_sleeping:
            HUDManager.draw_status(img, "SYSTEM ASLEEP : THUMBS DOWN TO WAKE", (0, 0, 255), scale=0.8)
            return True, is_dragging

        # --- 2. IDLE / CLUTCH (FIST) ---
        if sum(fingers) == 0 and not thumb_out:
            HUDManager.draw_status(img, "CLUTCH (FROZEN)", (0, 0, 255))
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            self.scroll_smoother.update(0) 
            self.command_mode_active = False # Reset intent
            return True, is_dragging

        # --- 3. COMMAND MODE (OPEN PALM + INTENT CHECK) ---
        if sum(fingers) == 4 and thumb_out:
            if self.palm_open_start_time == 0:
                self.palm_open_start_time = current_time
            
            # Check Intent: Have they held the palm open long enough?
            if current_time - self.palm_open_start_time > Config.Gestures.INTENT_DELAY:
                self.command_mode_active = True
                HUDManager.draw_status(img, "COMMAND MODE ACTIVE", (255, 100, 100))
                
                # Execute Swipes only if Command Mode is fully active
                if len(self.wrist_history_x) == self.wrist_history_x.maxlen:
                    dx = self.wrist_history_x[-1] - self.wrist_history_x[0]
                    dy = self.wrist_history_y[-1] - self.wrist_history_y[0]
                    
                    if current_time - self.cooldowns['window_swipe'] > Config.Gestures.COOLDOWN_DEFAULT:
                        if abs(dx) > abs(dy): # Horizontal Swipe
                            if dx > Config.Gestures.SWIPE_VELOCITY_THRESH:
                                pyautogui.hotkey('win', 'right')
                                self.cooldowns['window_swipe'] = current_time
                            elif dx < -Config.Gestures.SWIPE_VELOCITY_THRESH:
                                pyautogui.hotkey('win', 'left')
                                self.cooldowns['window_swipe'] = current_time
                        else: # Vertical Swipe
                            if dy > Config.Gestures.SWIPE_VELOCITY_THRESH: # Down
                                pyautogui.hotkey('win', 'd')
                                self.cooldowns['window_swipe'] = current_time
                            elif dy < -Config.Gestures.SWIPE_VELOCITY_THRESH: # Up
                                pyautogui.hotkey('win', 'up')
                                self.cooldowns['window_swipe'] = current_time
            else:
                # Loading Intent indicator
                HUDManager.draw_status(img, "HOLD TO ENGAGE...", (200, 200, 200))
            
            return True, is_dragging
        else:
            # Not an open palm, reset intent timer
            self.palm_open_start_time = 0
            self.command_mode_active = False


        # --- 4. ESCAPE KEY (Index + Middle Up, Thumb Out) ---
        if fingers == [1, 1, 0, 0] and thumb_out:
            HUDManager.draw_status(img, "ESCAPE", (255, 255, 255))
            if current_time - self.cooldowns['sys_keys'] > Config.Gestures.COOLDOWN_DEFAULT:
                pyautogui.press('esc')
                self.cooldowns['sys_keys'] = current_time
            return True, is_dragging

        # --- 5. WINDOWS KEY / START (Middle, Ring, Pinky Up) ---
        if fingers == [0, 1, 1, 1]:
            HUDManager.draw_status(img, "START MENU", (255, 255, 255))
            if current_time - self.cooldowns['sys_keys'] > Config.Gestures.COOLDOWN_DEFAULT:
                pyautogui.press('win')
                self.cooldowns['sys_keys'] = current_time
            return True, is_dragging

        # --- 6. VOLUME KNOB (Pinky Up, Thumb Out) ---
        if fingers == [0, 0, 0, 1] and thumb_out:
            HUDManager.draw_status(img, "AUDIO CONTROL", (0, 255, 255))
            if current_time - self.cooldowns['vol'] > Config.Gestures.COOLDOWN_FAST:
                if lm[4].y < lm[20].y: pyautogui.press('volumeup')
                else: pyautogui.press('volumedown')
                self.cooldowns['vol'] = current_time
            return True, is_dragging

        # --- 7. BRIGHTNESS KNOB (Index, Middle, Ring Up) ---
        if fingers == [1, 1, 1, 0] and HAS_SBC:
            HUDManager.draw_status(img, "BRIGHTNESS", (255, 255, 255))
            if current_time - self.cooldowns['bright'] > Config.Gestures.COOLDOWN_FAST:
                current_brightness = sbc.get_brightness(display=0)[0]
                if lm[8].y < lm[0].y - 0.2: sbc.set_brightness(min(100, current_brightness + 5))
                elif lm[8].y > lm[0].y - 0.1: sbc.set_brightness(max(0, current_brightness - 5))
                self.cooldowns['bright'] = current_time
            return True, is_dragging

        # --- 8. MEDIA CONTROLS (Index & Pinky Up - "Rock On") ---
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
                
                elif abs(dx) < 30 and (current_time - self.cooldowns['play'] > Config.Gestures.COOLDOWN_DEFAULT): 
                    pyautogui.press('playpause')
                    self.cooldowns['play'] = current_time
            return True, is_dragging

        # --- 9. AUTO SCROLL (Index + Middle Up) ---
        if fingers == [1, 1, 0, 0] and not thumb_out:
            HUDManager.draw_status(img, "AUTO SCROLL", (255, 255, 0))
            
            # Use middle finger base as anchor
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
                pyautogui.scroll(actual_speed)
                
            # Scroll visualizer
            center_h = int(h / 2)
            scroll_offset = int((actual_speed / Config.Gestures.SCROLL_MAX_SPEED) * 100)
            cv2.line(img, (50, center_h), (50, center_h - scroll_offset), (255, 255, 0), 10)
            
            if is_dragging: 
                pyautogui.mouseUp()
                is_dragging = False
            return True, is_dragging

        # Unhandled gesture, wind down scroll
        self.scroll_smoother.update(0)
        return False, is_dragging


class VisionOS:
    """
    Main Application Engine. 
    Handles camera I/O, AI processing loop, and hardware execution.
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_w, self.screen_h = pyautogui.size()
        
        self.smoother = DynamicSmoother()
        self.analyzer = HandAnalyzer()
        self.gestures = GestureController()
        
        self.dragging = False
        self.frozen_x, self.frozen_y = self.screen_w / 2, self.screen_h / 2
        self.is_frozen = False
        self.pinch_start_time = 0

    def run(self):
        print("VISION OS V6 PRO Online.")
        print("-> Edge Navigation Fixed: Full Screen Access Unlocked.")
        print("-> Intent Engine Active: Accidental swipes minimized.")
        print("-> THUMBS DOWN to Pause System.")
        
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success: break
            
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            HUDManager.draw_tracking_box(img, w, h)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = self.hands.process(img_rgb)
            img_rgb.flags.writeable = True
            
            # Failsafe: drop drag if hand leaves frame
            if not results.multi_hand_landmarks and self.dragging:
                pyautogui.mouseUp()
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

                # ==========================================
                # 1. PROCESS LEFT HAND (Mode 2 Clicker)
                # ==========================================
                if mode == 2 and left_hand and not self.gestures.is_sleeping:
                    self.mp_draw.draw_landmarks(img, left_hand, self.mp_hands.HAND_CONNECTIONS)
                    lm = left_hand.landmark
                    palm_w = MathUtils.get_dist(lm[5], lm[17], w, h)
                    
                    idx_pinch = (MathUtils.get_dist(lm[4], lm[8], w, h) / palm_w) < Config.Cursor.PINCH_RATIO
                    mid_pinch = (MathUtils.get_dist(lm[4], lm[12], w, h) / palm_w) < Config.Cursor.PINCH_RATIO
                    
                    if idx_pinch: left_click_held = True
                    if mid_pinch and not idx_pinch: right_click_triggered = True

                # ==========================================
                # 2. PROCESS ACTIVE HAND (Right / Navigation)
                # ==========================================
                if right_hand:
                    self.mp_draw.draw_landmarks(img, right_hand, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Offload raw tracking to Hysteresis Analyzer for stable binary outputs
                    fingers, thumb_out, lm = self.analyzer.analyze(right_hand, w, h)
                    
                    gesture_handled, self.dragging = self.gestures.process_system_gestures(
                        img, fingers, thumb_out, lm, w, h, self.dragging
                    )

                    # --- NAVIGATION & STRICT CLICKS (If no system gesture is active) ---
                    if not gesture_handled and not self.gestures.is_sleeping:
                        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                        
                        # V6 UPGRADE: Numpy Interp automatically clamps values. 
                        # Reaching the tracking box boundary = reaching screen edge!
                        mapped_x = np.interp(ix, [Config.Tracking.FRAME_MARGIN_X, w - Config.Tracking.FRAME_MARGIN_X], [0, self.screen_w])
                        mapped_y = np.interp(iy, [Config.Tracking.FRAME_MARGIN_Y, h - Config.Tracking.FRAME_MARGIN_Y], [0, self.screen_h])

                        if mode == 2:
                            if left_click_held or right_click_triggered:
                                if not self.is_frozen:
                                    self.frozen_x, self.frozen_y = self.smoother.prev_x, self.smoother.prev_y
                                    self.is_frozen = True
                                cursor_x, cursor_y = self.frozen_x, self.frozen_y
                                HUDManager.draw_cursor(img, ix, iy, color=(0, 0, 255), radius=15)
                            else:
                                self.is_frozen = False
                                cursor_x, cursor_y = self.smoother.update(mapped_x, mapped_y)
                                HUDManager.draw_cursor(img, ix, iy, color=(255, 255, 0), radius=10)

                        elif mode == 1:
                            palm_w = MathUtils.get_dist(lm[5], lm[17], w, h)
                            
                            idx_pinch = (MathUtils.get_dist(lm[4], lm[8], w, h) / palm_w) < Config.Cursor.PINCH_RATIO
                            mid_pinch = (MathUtils.get_dist(lm[4], lm[12], w, h) / palm_w) < Config.Cursor.PINCH_RATIO
                            
                            if idx_pinch:
                                left_click_held = True
                                if not self.dragging:
                                    self.pinch_start_time = time.time()
                                if time.time() - self.pinch_start_time < Config.Cursor.CLICK_FREEZE_TIME and self.smoother.prev_x is not None:
                                    mapped_x, mapped_y = self.smoother.prev_x, self.smoother.prev_y
                                HUDManager.draw_cursor(img, ix, iy, color=(0, 255, 0), radius=15)
                            else:
                                HUDManager.draw_cursor(img, ix, iy, color=(255, 255, 0), radius=10)
                            
                            if mid_pinch and not idx_pinch: right_click_triggered = True

                            cursor_x, cursor_y = self.smoother.update(mapped_x, mapped_y)

                        try:
                            pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
                        except pyautogui.FailSafeException:
                            pass 

            # ==========================================
            # 3. GLOBAL CLICK EXECUTION
            # ==========================================
            if not self.gestures.is_sleeping:
                if left_click_held:
                    HUDManager.draw_status(img, f"LEFT CLICK ({'LEFT HAND' if mode == 2 else '1-HAND'})", 
                                           (0, 255, 0), (20, 110), 0.8)
                    if not self.dragging:
                        pyautogui.mouseDown()
                        self.dragging = True
                else:
                    if self.dragging:
                        pyautogui.mouseUp()
                        self.dragging = False

                if right_click_triggered and (time.time() - self.gestures.cooldowns['right_click'] > 0.5):
                    HUDManager.draw_status(img, "RIGHT CLICK", (0, 0, 255), (20, 150), 0.8)
                    pyautogui.rightClick()
                    self.gestures.cooldowns['right_click'] = time.time()

            cv2.imshow("VISION OS V6 PRO", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VisionOS()
    app.run()