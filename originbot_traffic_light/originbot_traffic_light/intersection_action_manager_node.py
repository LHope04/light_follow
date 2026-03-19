import enum
import math
from collections import deque
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
from originbot_msgs.msg import MovingParam, TrafficDecision

_DECISION_TO_MOVING = {
    TrafficDecision.STRAIGHT: MovingParam.MOVING_FORWARD,
    TrafficDecision.LEFT: MovingParam.MOVING_LEFT,  # 绝不搞错，标准的 3！
    TrafficDecision.RIGHT: MovingParam.MOVING_RIGHT,
    TrafficDecision.STOP: MovingParam.MOVING_STOP,
}

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

class IntersectionActionManager(Node):
    def __init__(self):
        super().__init__('intersection_action_manager')
        # ==========================================
        # 究极平衡：8% 的面积识别率 + 连续 2 帧防抖
        # ==========================================
        self.declare_parameter('conf_th', 0.08)     
        self.declare_parameter('stable_frames', 3) 
        
        self.declare_parameter('execute_timeout', 10.0)
        self.declare_parameter('done_reset_delay', 0.5)
        self.declare_parameter('max_vel_execute', 0.1) 
        self.declare_parameter('turn_target_angle_deg', 90.0)
        self.declare_parameter('turn_max_angle_deg', 120.0)
        self.declare_parameter('turn_min_angle_deg_for_line_exit', 45.0) 
        self.declare_parameter('right_turn_negative', True)
        self.declare_parameter('turn_use_line_reacquire', True)
        
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

        self.create_subscription(TrafficDecision, '/detect/traffic_decision', self._on_traffic_decision, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 30)
        self.create_subscription(Bool, '/line_follow/line_found', self._on_line_found, 30)
        self.create_subscription(Bool, '/line_follow/crossroad', self._on_crossroad, 10)

        self._pub_moving_state = self.create_publisher(MovingParam, '/control/moving/state', 10)
        self._pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)

        self._state = _State.IDLE
        self._state_start_time = None
        self._clear_start_time = None
        self._decision_buffer = deque(maxlen=self._stable_frames)
        self._locked_decision = TrafficDecision.UNKNOWN  
        self._at_crossroad = False                       

        self._current_yaw = None
        self._turn_start_yaw = None
        self._turn_direction = TrafficDecision.UNKNOWN
        self._line_true_count = 0

        self.create_timer(0.05, self._tick)

    def _on_crossroad(self, msg: Bool):
        self._at_crossroad = msg.data

    def _on_traffic_decision(self, msg: TrafficDecision):
        filtered = msg.state if msg.confidence >= self._conf_th else TrafficDecision.UNKNOWN
        self._decision_buffer.append(filtered)

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self._current_yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

    def _on_line_found(self, msg: Bool):
        if msg.data:
            self._line_true_count += 1
        else:
            self._line_true_count = 0

    def _get_stable_decision(self):
        if len(self._decision_buffer) == self._stable_frames:
            recent = list(self._decision_buffer)
            if all(s == recent[0] for s in recent):
                return recent[0]
        return TrafficDecision.UNKNOWN

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
            if candidate in (TrafficDecision.LEFT, TrafficDecision.RIGHT, TrafficDecision.STRAIGHT):
                self._locked_decision = candidate
                self.get_logger().info(f'Command received: {candidate}.')
                if candidate == TrafficDecision.STRAIGHT:
                    self._transition(_State.CROSSROAD_CLEARING)
                else:
                    self._transition(_State.EXECUTE_TURN)

        elif self._state == _State.EXECUTE_TURN:
            if self._turn_direction in (TrafficDecision.LEFT, TrafficDecision.RIGHT):
                if self._current_yaw is not None and self._turn_start_yaw is not None:
                    turned_deg = abs(math.degrees(_normalize_angle(self._current_yaw - self._turn_start_yaw)))
                    if self._turn_use_line_reacquire and turned_deg >= self._turn_min_angle_deg_for_line_exit and self._line_true_count >= 2:
                        self.get_logger().info(f'Found new line at {turned_deg:.1f} deg.')
                        self._transition(_State.CROSSROAD_CLEARING)
                        return
                    if turned_deg >= self._turn_max_angle_deg:
                        self._transition(_State.CROSSROAD_CLEARING)
                        return
            
            if self._state_start_time is not None and (now - self._state_start_time).nanoseconds * 1e-9 > self._execute_timeout:
                self._transition(_State.CROSSROAD_CLEARING)

        elif self._state == _State.CROSSROAD_CLEARING:
            if not self._at_crossroad:
                if self._clear_start_time is None:
                    self._clear_start_time = now
                elif (now - self._clear_start_time).nanoseconds * 1e-9 > 1.0:
                    self._transition(_State.DONE)
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
        now = self.get_clock().now()
        self._state_start_time = now  

        msg = MovingParam()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'base_link'

        if new_state == _State.CROSSROAD_APPROACH:
            msg.moving_type = MovingParam.MOVING_UNKNOWN 
            msg.moving_value = 0.0
            self._pub_moving_state.publish(msg)

        elif new_state == _State.CROSSROAD_STOP:
            self._decision_buffer.clear()
            msg.moving_type = MovingParam.MOVING_STOP
            msg.moving_value = 0.0
            self._pub_moving_state.publish(msg)
            self._publish_max_vel(0.0)

        elif new_state == _State.EXECUTE_TURN:
            self._turn_direction = self._locked_decision
            self._turn_start_yaw = self._current_yaw
            self._line_true_count = 0
            msg.moving_type = _DECISION_TO_MOVING.get(self._turn_direction, MovingParam.MOVING_UNKNOWN)
            
            if self._turn_direction == TrafficDecision.LEFT:
                msg.moving_value = math.radians(self._turn_target_angle_deg)
            elif self._turn_direction == TrafficDecision.RIGHT:
                angle = math.radians(self._turn_target_angle_deg)
                msg.moving_value = -angle if self._right_turn_negative else angle
            else:
                msg.moving_value = 0.0
            self._pub_moving_state.publish(msg)
            self._publish_max_vel(self._max_vel_execute)

        elif new_state == _State.CROSSROAD_CLEARING:
            self._clear_start_time = None
            msg.moving_type = MovingParam.MOVING_FORWARD
            msg.moving_value = 0.0
            self._pub_moving_state.publish(msg)
            self._publish_max_vel(0.12) 

        elif new_state == _State.DONE:
            msg.moving_type = MovingParam.MOVING_STOP
            msg.moving_value = 0.0
            self._pub_moving_state.publish(msg)
            self._publish_max_vel(0.0)

        elif new_state == _State.IDLE:
            self._locked_decision = TrafficDecision.UNKNOWN
            self._decision_buffer.clear()
            msg.moving_type = MovingParam.MOVING_UNKNOWN
            msg.moving_value = 0.0
            self._pub_moving_state.publish(msg)

        self.get_logger().info(f'State: {old_state.value} -> {new_state.value}')

    def _publish_max_vel(self, vel: float):
        msg = Float64()
        msg.data = float(vel)
        self._pub_max_vel.publish(msg)

def main():
    rclpy.init()
    node = IntersectionActionManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()