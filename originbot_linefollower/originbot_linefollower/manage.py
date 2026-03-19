import cv2
import math
import enum
import numpy as np
from collections import deque
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from originbot_msgs.msg import TrafficLight

from originbot_linefollower._utils import DebounceFilter as _DebounceFilter
from originbot_linefollower._utils import clamp_roi as _clamp_roi
from originbot_linefollower._utils import mask_ratio as _mask_ratio
from originbot_linefollower._utils import ratio_to_roi as _ratio_to_roi

class _State(enum.Enum):
    IDLE = 'IDLE'
    CROSSROAD_APPROACH = 'CROSSROAD_APPROACH'
    CROSSROAD_STOP = 'CROSSROAD_STOP'
    EXECUTE_TURN = 'EXECUTE_TURN'
    CROSSROAD_CLEARING = 'CROSSROAD_CLEARING'
    DONE = 'DONE'

def _quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def _normalize_angle(rad):
    while rad > math.pi: rad -= 2.0 * math.pi
    while rad < -math.pi: rad += 2.0 * math.pi
    return rad

class TrafficManager(Node):
    def __init__(self):
        super().__init__('traffic_manager')
        
        # --- 1. 视觉参数声明 ---
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_topic', '/traffic_light/state')
        self.declare_parameter('mode', 'roi')
        self.declare_parameter('debug', True)
        self.declare_parameter('debug_topic', '/traffic_light/debug')

        self.declare_parameter('left_x', 128); self.declare_parameter('left_y', 154)
        self.declare_parameter('left_w', 131); self.declare_parameter('left_h', 109)
        self.declare_parameter('mid_x', 259); self.declare_parameter('mid_y', 154)
        self.declare_parameter('mid_w', 131); self.declare_parameter('mid_h', 109)
        self.declare_parameter('right_x', 391); self.declare_parameter('right_y', 154)
        self.declare_parameter('right_w', 131); self.declare_parameter('right_h', 109)

        for _prefix in ('left', 'mid', 'right'):
            self.declare_parameter(f'{_prefix}_x_ratio', 0.0)
            self.declare_parameter(f'{_prefix}_y_ratio', 0.0)
            self.declare_parameter(f'{_prefix}_w_ratio', 0.0)
            self.declare_parameter(f'{_prefix}_h_ratio', 0.0)

        self.declare_parameter('green_h_low', 35); self.declare_parameter('green_s_low', 80); self.declare_parameter('green_v_low', 80)
        self.declare_parameter('green_h_high', 90); self.declare_parameter('green_s_high', 255); self.declare_parameter('green_v_high', 255)
        self.declare_parameter('red1_h_low', 0); self.declare_parameter('red1_s_low', 80); self.declare_parameter('red1_v_low', 80)
        self.declare_parameter('red1_h_high', 10); self.declare_parameter('red1_s_high', 255); self.declare_parameter('red1_v_high', 255)
        self.declare_parameter('red2_h_low', 160); self.declare_parameter('red2_s_low', 80); self.declare_parameter('red2_v_low', 80)
        self.declare_parameter('red2_h_high', 179); self.declare_parameter('red2_s_high', 255); self.declare_parameter('red2_v_high', 255)

        self.declare_parameter('min_green_ratio', 0.01)
        self.declare_parameter('min_red_ratio', 0.01)
        self.declare_parameter('publish_unknown', True)
        self.declare_parameter('debounce_frames', 3)
        self.declare_parameter('debounce_unknown', True)
        
        # --- 2. 动作策略参数声明 ---
        self.declare_parameter('conf_th', 0.6)     
        self.declare_parameter('stable_frames', 3) 
        self.declare_parameter('execute_timeout', 10.0)
        self.declare_parameter('done_reset_delay', 3.0)
        self.declare_parameter('max_vel_execute', 0.15) 
        self.declare_parameter('turn_target_angle_deg', 90.0)
        self.declare_parameter('turn_max_angle_deg', 105.0)
        self.declare_parameter('turn_min_angle_deg_for_line_exit', 55.0) 
        self.declare_parameter('right_turn_negative', True)
        self.declare_parameter('turn_use_line_reacquire', True)

        # 参数提取
        self.green_low = (int(self.get_parameter('green_h_low').value), int(self.get_parameter('green_s_low').value), int(self.get_parameter('green_v_low').value))
        self.green_high = (int(self.get_parameter('green_h_high').value), int(self.get_parameter('green_s_high').value), int(self.get_parameter('green_v_high').value))
        self.red1_low = (int(self.get_parameter('red1_h_low').value), int(self.get_parameter('red1_s_low').value), int(self.get_parameter('red1_v_low').value))
        self.red1_high = (int(self.get_parameter('red1_h_high').value), int(self.get_parameter('red1_s_high').value), int(self.get_parameter('red1_v_high').value))
        self.red2_low = (int(self.get_parameter('red2_h_low').value), int(self.get_parameter('red2_s_low').value), int(self.get_parameter('red2_v_low').value))
        self.red2_high = (int(self.get_parameter('red2_h_high').value), int(self.get_parameter('red2_s_high').value), int(self.get_parameter('red2_v_high').value))
        self.min_green = float(self.get_parameter('min_green_ratio').value)
        self.min_red = float(self.get_parameter('min_red_ratio').value)

        self._conf_th = float(self.get_parameter('conf_th').value)
        self._stable_frames = int(self.get_parameter('stable_frames').value)
        self._execute_timeout = float(self.get_parameter('execute_timeout').value)
        self._done_reset_delay = float(self.get_parameter('done_reset_delay').value)
        self._max_vel_execute = float(self.get_parameter('max_vel_execute').value)
        self._turn_target_angle_deg = float(self.get_parameter('turn_target_angle_deg').value)
        self._turn_max_angle_deg = float(self.get_parameter('turn_max_angle_deg').value)
        self._turn_min_angle_deg_for_line_exit = float(self.get_parameter('turn_min_angle_deg_for_line_exit').value)
        self._right_turn_negative = bool(self.get_parameter('right_turn_negative').value)
        self._turn_use_line_reacquire = bool(self.get_parameter('turn_use_line_reacquire').value)

        self.bridge = CvBridge()
        self.publish_unknown = bool(self.get_parameter('publish_unknown').value)
        self.mode = self.get_parameter('mode').value
        self.state_names = {TrafficLight.LEFT: "LEFT", TrafficLight.STRAIGHT: "STRAIGHT", TrafficLight.RIGHT: "RIGHT", TrafficLight.STOP: "STOP", TrafficLight.UNKNOWN: "UNKNOWN"}
        self._debounce = _DebounceFilter(debounce_frames=int(self.get_parameter('debounce_frames').value), debounce_unknown=bool(self.get_parameter('debounce_unknown').value), unknown_state=TrafficLight.UNKNOWN)
        self._ratio_warned = set()

        # 内部系统状态变量
        self._state = _State.IDLE
        self._state_start_time = None
        self._clear_start_time = None
        self._decision_buffer = deque(maxlen=self._stable_frames)
        self._locked_decision = TrafficLight.UNKNOWN  
        self._at_crossroad = False                       
        self._current_yaw = None
        self._turn_start_yaw = None
        self._turn_direction = TrafficLight.UNKNOWN
        self._line_true_count = 0
        
        # 内部运动适配变量
        self._current_moving_type = 0  # 0=IDLE/LINE_FOLLOW, 1=STOP, 2=FWD, 3=LEFT, 4=RIGHT
        self._current_max_vel = 0.0

        # --- 3. ROS 通信 ---
        self.sub_img = self.create_subscription(Image, self.get_parameter('image_topic').value, self._on_image, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self._on_odom, 30)
        self.sub_line = self.create_subscription(Bool, '/line_follow/line_found', self._on_line_found, 30)
        self.sub_cross = self.create_subscription(Bool, '/line_follow/crossroad', self._on_crossroad, 10)
        self.sub_cmd = self.create_subscription(Twist, '/line_follow/cmd_vel', self._on_line_cmd, 10)

        self.pub_tl = self.create_publisher(TrafficLight, self.get_parameter('output_topic').value, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.debug_pub = None
        if self.get_parameter('debug').value:
            self.debug_pub = self.create_publisher(Image, self.get_parameter('debug_topic').value, 10)

        self.create_timer(0.05, self._tick)
        self.get_logger().info('Unified Traffic Manager Initialized.')

    # ==========================================
    # 视觉检测逻辑
    # ==========================================
    def _roi_params(self, prefix, img_w, img_h):
        xr = float(self.get_parameter(f'{prefix}_x_ratio').value)
        yr = float(self.get_parameter(f'{prefix}_y_ratio').value)
        wr = float(self.get_parameter(f'{prefix}_w_ratio').value)
        hr = float(self.get_parameter(f'{prefix}_h_ratio').value)
        if wr > 0.0 or hr > 0.0:
            result = _ratio_to_roi(xr, yr, wr, hr, img_w, img_h)
            if result is not None: return result
            if prefix not in self._ratio_warned: self._ratio_warned.add(prefix)
        x = self.get_parameter(f'{prefix}_x').value; y = self.get_parameter(f'{prefix}_y').value
        w = self.get_parameter(f'{prefix}_w').value; h = self.get_parameter(f'{prefix}_h').value
        return _clamp_roi(x, y, w, h, img_w, img_h)

    def _eval_zone(self, hsv, prefix, img_w, img_h):
        x, y, w, h = self._roi_params(prefix, img_w, img_h)
        zone = hsv[y:y + h, x:x + w]
        green_mask = cv2.inRange(zone, self.green_low, self.green_high)
        red_mask = cv2.bitwise_or(cv2.inRange(zone, self.red1_low, self.red1_high), cv2.inRange(zone, self.red2_low, self.red2_high))
        return _mask_ratio(green_mask), _mask_ratio(red_mask)

    def _publish_debug(self, bgr, msg_header, state=None):
        if self.debug_pub is None: return
        dbg = bgr.copy()
        if state is not None:
            state_name = self.state_names.get(state, "UNKNOWN")
            text_color = (0, 0, 0)
            if state == TrafficLight.STOP: text_color = (0, 0, 255) 
            elif state == TrafficLight.LEFT: text_color = (255, 255, 0) 
            elif state == TrafficLight.RIGHT: text_color = (0, 255, 255) 
            elif state == TrafficLight.STRAIGHT: text_color = (0, 255, 0) 
            cv2.putText(dbg, f"STATE: {state_name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            
        if self.mode == 'roi':
            img_h, img_w = dbg.shape[:2]
            lx, ly, lw, lh = self._roi_params('left', img_w, img_h)
            mx, my, mw, mh = self._roi_params('mid', img_w, img_h)
            rx, ry, rw, rh = self._roi_params('right', img_w, img_h)
            cv2.rectangle(dbg, (lx, ly), (lx+lw, ly+lh), (255, 0, 0), 2); cv2.putText(dbg, "L", (lx, ly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.rectangle(dbg, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2); cv2.putText(dbg, "M", (mx, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.rectangle(dbg, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2); cv2.putText(dbg, "R", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            debug_msg.header = msg_header
            self.debug_pub.publish(debug_msg)
        except Exception: pass

    def _on_image(self, msg):
        try: bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception: return

        out = TrafficLight()
        out.header = msg.header
        if self.mode != 'card': self._process_roi_mode(bgr, out)
        out.state, out.confidence = self._debounce.update(out.state, out.confidence)

        if self.publish_unknown or out.state != TrafficLight.UNKNOWN:
            self.pub_tl.publish(out)
            # 内部抛送给决策缓冲区
            filtered = out.state if out.confidence >= self._conf_th else TrafficLight.UNKNOWN
            self._decision_buffer.append(filtered)

    def _process_roi_mode(self, bgr, out):
        img_h, img_w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        g_l, r_l = self._eval_zone(hsv, 'left', img_w, img_h)
        g_m, r_m = self._eval_zone(hsv, 'mid', img_w, img_h)
        g_r, r_r = self._eval_zone(hsv, 'right', img_w, img_h)

        greens = [g_l, g_m, g_r]; reds = [r_l, r_m, r_r]
        green_on = [g > self.min_green for g in greens]; red_on = [r > self.min_red for r in reds]

        if sum(green_on) == 1:
            out.confidence = float(max(greens))
            if green_on[0]: out.state = TrafficLight.LEFT
            elif green_on[1]: out.state = TrafficLight.STRAIGHT
            else: out.state = TrafficLight.RIGHT
        elif sum(green_on) == 0 and all(red_on):
            out.state = TrafficLight.STOP
            out.confidence = float(min(reds))
        else:
            out.state = TrafficLight.UNKNOWN
            out.confidence = 0.0

        self._publish_debug(bgr, out.header, out.state) 

    # ==========================================
    # 动作管理器逻辑
    # ==========================================
    def _on_crossroad(self, msg: Bool): self._at_crossroad = msg.data
    def _on_odom(self, msg: Odometry): self._current_yaw = _quat_to_yaw(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
    def _on_line_found(self, msg: Bool): 
        if msg.data: self._line_true_count += 1
        else: self._line_true_count = 0

    def _get_stable_decision(self):
        if len(self._decision_buffer) == self._stable_frames:
            recent = list(self._decision_buffer)
            if all(s == recent[0] for s in recent): return recent[0]
        return TrafficLight.UNKNOWN

    def _tick(self):
        now = self.get_clock().now()
        if self._state == _State.IDLE:
            if self._at_crossroad:
                self.get_logger().info('Crossroad detected. Approaching center for 1s.')
                self._transition(_State.CROSSROAD_APPROACH)
        elif self._state == _State.CROSSROAD_APPROACH:
            if self._state_start_time is not None and (now - self._state_start_time).nanoseconds * 1e-9 > 1.0:
                self.get_logger().info('Reached center. Stopping and waiting INDEFINITELY for signal.')
                self._transition(_State.CROSSROAD_STOP)
        elif self._state == _State.CROSSROAD_STOP:
            candidate = self._get_stable_decision()
            if candidate in (TrafficLight.LEFT, TrafficLight.RIGHT, TrafficLight.STRAIGHT):
                self._locked_decision = candidate
                self.get_logger().info(f'Command received: {candidate}.')
                if candidate == TrafficLight.STRAIGHT: self._transition(_State.CROSSROAD_CLEARING)
                else: self._transition(_State.EXECUTE_TURN)
        elif self._state == _State.EXECUTE_TURN:
            if self._turn_direction in (TrafficLight.LEFT, TrafficLight.RIGHT):
                if self._current_yaw is not None and self._turn_start_yaw is not None:
                    turned_deg = abs(math.degrees(_normalize_angle(self._current_yaw - self._turn_start_yaw)))
                    if self._turn_use_line_reacquire and turned_deg >= self._turn_min_angle_deg_for_line_exit and self._line_true_count >= 2:
                        self.get_logger().info(f'Found new line at {turned_deg:.1f} deg.')
                        self._transition(_State.CROSSROAD_CLEARING); return
                    if turned_deg >= self._turn_max_angle_deg:
                        self._transition(_State.CROSSROAD_CLEARING); return
            if self._state_start_time is not None and (now - self._state_start_time).nanoseconds * 1e-9 > self._execute_timeout:
                self._transition(_State.CROSSROAD_CLEARING)
        elif self._state == _State.CROSSROAD_CLEARING:
            if not self._at_crossroad:
                if self._clear_start_time is None: self._clear_start_time = now
                elif (now - self._clear_start_time).nanoseconds * 1e-9 > 1.0: self._transition(_State.DONE)
            else:
                self._clear_start_time = None
            if self._state_start_time is not None and (now - self._state_start_time).nanoseconds * 1e-9 > 5.0:
                self._transition(_State.DONE)
        elif self._state == _State.DONE:
            if self._state_start_time is not None and (now - self._state_start_time).nanoseconds * 1e-9 > self._done_reset_delay:
                self._transition(_State.IDLE)

    def _transition(self, new_state: _State):
        old_state = self._state
        self._state = new_state
        self._state_start_time = self.get_clock().now()

        if new_state == _State.CROSSROAD_APPROACH:
            self._current_moving_type = 0
            self._current_max_vel = 0.0
        elif new_state == _State.CROSSROAD_STOP:
            self._decision_buffer.clear()
            self._current_moving_type = 1 
            self._current_max_vel = 0.0
        elif new_state == _State.EXECUTE_TURN:
            self._turn_direction = self._locked_decision
            self._turn_start_yaw = self._current_yaw
            self._line_true_count = 0
            
            if self._turn_direction == TrafficLight.LEFT: self._current_moving_type = 3
            elif self._turn_direction == TrafficLight.RIGHT: self._current_moving_type = 4
            elif self._turn_direction == TrafficLight.STRAIGHT: self._current_moving_type = 2
            else: self._current_moving_type = 0
            self._current_max_vel = self._max_vel_execute
        elif new_state == _State.CROSSROAD_CLEARING:
            self._clear_start_time = None
            self._current_moving_type = 2
            self._current_max_vel = 0.12 
        elif new_state == _State.DONE:
            self._current_moving_type = 1
            self._current_max_vel = 0.0
        elif new_state == _State.IDLE:
            self._locked_decision = TrafficLight.UNKNOWN
            self._decision_buffer.clear()
            self._current_moving_type = 0
            self._current_max_vel = 0.0

        self._publish_action_cmd()
        self.get_logger().info(f'State: {old_state.value} -> {new_state.value}')

    # ==========================================
    # 运动指令适配器逻辑
    # ==========================================
    def _on_line_cmd(self, msg: Twist):
        # 只要系统处于 IDLE 状态，就将底盘控制权完全交给巡线逻辑
        if self._current_moving_type == 0:
            self.pub_cmd_vel.publish(msg)

    def _publish_action_cmd(self):
        if self._current_moving_type == 0: return # 0 表示非控制状态，交给巡线
        cmd = Twist()
        if self._current_moving_type == 1: pass # STOP: 0.0, 0.0
        elif self._current_moving_type == 2: cmd.linear.x = self._current_max_vel # FWD
        elif self._current_moving_type == 3: cmd.angular.z = 0.3 # LEFT
        elif self._current_moving_type == 4: cmd.angular.z = -0.3 # RIGHT
        self.pub_cmd_vel.publish(cmd)

def main():
    rclpy.init()
    node = TrafficManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()