import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from originbot_msgs.msg import TrafficDecision
from originbot_msgs.msg import TrafficLight
from rclpy.node import Node
from sensor_msgs.msg import Image

from originbot_traffic_light._utils import DebounceFilter as _DebounceFilter
from originbot_traffic_light._utils import clamp_roi as _clamp_roi
from originbot_traffic_light._utils import mask_ratio as _mask_ratio
from originbot_traffic_light._utils import order_corners as _order_corners
from originbot_traffic_light._utils import ratio_to_roi as _ratio_to_roi

_DEBUG_THUMB_MAX_DIM = 160

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_topic', '/traffic_light/state')
        self.declare_parameter('decision_topic', '/detect/traffic_decision')
        self.declare_parameter('mode', 'roi')

        self.declare_parameter('left_x', 128)
        self.declare_parameter('left_y', 154)
        self.declare_parameter('left_w', 131)
        self.declare_parameter('left_h', 109)
        self.declare_parameter('mid_x', 259)
        self.declare_parameter('mid_y', 154)
        self.declare_parameter('mid_w', 131)
        self.declare_parameter('mid_h', 109)
        self.declare_parameter('right_x', 391)
        self.declare_parameter('right_y', 154)
        self.declare_parameter('right_w', 131)
        self.declare_parameter('right_h', 109)

        for _prefix in ('left', 'mid', 'right'):
            self.declare_parameter(f'{_prefix}_x_ratio', 0.0)
            self.declare_parameter(f'{_prefix}_y_ratio', 0.0)
            self.declare_parameter(f'{_prefix}_w_ratio', 0.0)
            self.declare_parameter(f'{_prefix}_h_ratio', 0.0)

        self.declare_parameter('green_h_low', 35)
        self.declare_parameter('green_s_low', 80)
        self.declare_parameter('green_v_low', 80)
        self.declare_parameter('green_h_high', 90)
        self.declare_parameter('green_s_high', 255)
        self.declare_parameter('green_v_high', 255)
        self.declare_parameter('red1_h_low', 0)
        self.declare_parameter('red1_s_low', 80)
        self.declare_parameter('red1_v_low', 80)
        self.declare_parameter('red1_h_high', 10)
        self.declare_parameter('red1_s_high', 255)
        self.declare_parameter('red1_v_high', 255)
        self.declare_parameter('red2_h_low', 160)
        self.declare_parameter('red2_s_low', 80)
        self.declare_parameter('red2_v_low', 80)
        self.declare_parameter('red2_h_high', 179)
        self.declare_parameter('red2_s_high', 255)
        self.declare_parameter('red2_v_high', 255)

        self.declare_parameter('min_green_ratio', 0.01)
        self.declare_parameter('min_red_ratio', 0.01)
        self.declare_parameter('card_warp_width', 320)
        self.declare_parameter('card_warp_height', 120)
        self.declare_parameter('min_card_area', 5000)
        self.declare_parameter('card_aspect_ratio_min', 1.5)
        self.declare_parameter('card_aspect_ratio_max', 6.0)
        self.declare_parameter('band_margin_px', 5)
        self.declare_parameter('publish_unknown', True)
        self.declare_parameter('debounce_frames', 3)
        self.declare_parameter('debounce_unknown', True)
        
        self.declare_parameter('debug', True)
        self.declare_parameter('debug_topic', '/traffic_light/debug')

        self.green_low = (int(self.get_parameter('green_h_low').value), int(self.get_parameter('green_s_low').value), int(self.get_parameter('green_v_low').value))
        self.green_high = (int(self.get_parameter('green_h_high').value), int(self.get_parameter('green_s_high').value), int(self.get_parameter('green_v_high').value))
        self.red1_low = (int(self.get_parameter('red1_h_low').value), int(self.get_parameter('red1_s_low').value), int(self.get_parameter('red1_v_low').value))
        self.red1_high = (int(self.get_parameter('red1_h_high').value), int(self.get_parameter('red1_s_high').value), int(self.get_parameter('red1_v_high').value))
        self.red2_low = (int(self.get_parameter('red2_h_low').value), int(self.get_parameter('red2_s_low').value), int(self.get_parameter('red2_v_low').value))
        self.red2_high = (int(self.get_parameter('red2_h_high').value), int(self.get_parameter('red2_s_high').value), int(self.get_parameter('red2_v_high').value))
        self.min_green = float(self.get_parameter('min_green_ratio').value)
        self.min_red = float(self.get_parameter('min_red_ratio').value)

        self.bridge = CvBridge()
        self.publish_unknown = bool(self.get_parameter('publish_unknown').value)
        
        self.state_names = {
            TrafficLight.LEFT: "LEFT",
            TrafficLight.STRAIGHT: "STRAIGHT",
            TrafficLight.RIGHT: "RIGHT",
            TrafficLight.STOP: "STOP",
            TrafficLight.UNKNOWN: "UNKNOWN"
        }

        self._debounce = _DebounceFilter(
            debounce_frames=int(self.get_parameter('debounce_frames').value),
            debounce_unknown=bool(self.get_parameter('debounce_unknown').value),
            unknown_state=TrafficLight.UNKNOWN,
        )

        self._ratio_warned = set()
        self.mode = self.get_parameter('mode').value

        self.sub = self.create_subscription(Image, self.get_parameter('image_topic').value, self._on_image, 10)
        self.pub = self.create_publisher(TrafficLight, self.get_parameter('output_topic').value, 10)
        self.decision_pub = self.create_publisher(TrafficDecision, self.get_parameter('decision_topic').value, 10)

        self.debug_pub = None
        if self.get_parameter('debug').value:
            self.debug_pub = self.create_publisher(Image, self.get_parameter('debug_topic').value, 10)

    def _roi_params(self, prefix, img_w, img_h):
        xr = float(self.get_parameter(f'{prefix}_x_ratio').value)
        yr = float(self.get_parameter(f'{prefix}_y_ratio').value)
        wr = float(self.get_parameter(f'{prefix}_w_ratio').value)
        hr = float(self.get_parameter(f'{prefix}_h_ratio').value)
        if wr > 0.0 or hr > 0.0:
            result = _ratio_to_roi(xr, yr, wr, hr, img_w, img_h)
            if result is not None: return result
            if prefix not in self._ratio_warned: self._ratio_warned.add(prefix)
        x = self.get_parameter(f'{prefix}_x').value
        y = self.get_parameter(f'{prefix}_y').value
        w = self.get_parameter(f'{prefix}_w').value
        h = self.get_parameter(f'{prefix}_h').value
        return _clamp_roi(x, y, w, h, img_w, img_h)

    def _eval_zone(self, hsv, prefix, img_w, img_h):
        x, y, w, h = self._roi_params(prefix, img_w, img_h)
        zone = hsv[y:y + h, x:x + w]
        green_mask = cv2.inRange(zone, self.green_low, self.green_high)
        red_mask = cv2.bitwise_or(
            cv2.inRange(zone, self.red1_low, self.red1_high),
            cv2.inRange(zone, self.red2_low, self.red2_high),
        )
        return _mask_ratio(green_mask), _mask_ratio(red_mask)

    def _publish_debug(self, bgr, corners, warped, msg_header, state=None):
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
            
            cv2.rectangle(dbg, (lx, ly), (lx+lw, ly+lh), (255, 0, 0), 2)
            cv2.rectangle(dbg, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)
            cv2.rectangle(dbg, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
            
            cv2.putText(dbg, "L", (lx, ly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(dbg, "M", (mx, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(dbg, "R", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            debug_msg.header = msg_header
            self.debug_pub.publish(debug_msg)
        except Exception:
            pass

    def _on_image(self, msg):
        try: bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception: return

        out = TrafficLight()
        out.header = msg.header

        if self.mode == 'card': pass 
        else: self._process_roi_mode(bgr, out)

        out.state, out.confidence = self._debounce.update(out.state, out.confidence)

        if self.publish_unknown or out.state != TrafficLight.UNKNOWN:
            self.pub.publish(out)
            decision = TrafficDecision()
            decision.header = out.header
            decision.state = out.state
            decision.confidence = out.confidence
            self.decision_pub.publish(decision)

    def _process_roi_mode(self, bgr, out):
        img_h, img_w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        g_l, r_l = self._eval_zone(hsv, 'left', img_w, img_h)
        g_m, r_m = self._eval_zone(hsv, 'mid', img_w, img_h)
        g_r, r_r = self._eval_zone(hsv, 'right', img_w, img_h)

        greens = [g_l, g_m, g_r]
        reds = [r_l, r_m, r_r]
        green_on = [g > self.min_green for g in greens]
        red_on = [r > self.min_red for r in reds]

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

        self._publish_debug(bgr, None, None, out.header, out.state) 

def main():
    rclpy.init()
    node = TrafficLightDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()