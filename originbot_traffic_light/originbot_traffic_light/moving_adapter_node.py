import rclpy
from rclpy.node import Node
from originbot_msgs.msg import MovingParam
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

class MovingAdapter(Node):
    def __init__(self):
        super().__init__('moving_adapter')
        self.sub_moving_state = self.create_subscription(
            MovingParam, '/control/moving/state', self._on_moving_state, 10)
        self.sub_max_vel = self.create_subscription(
            Float64, '/control/max_vel', self._on_max_vel, 10)
        self.sub_line_cmd = self.create_subscription(
            Twist, '/line_follow/cmd_vel', self._on_line_cmd, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self._current_moving_type = 0  
        self._current_max_vel = 0.0
        self._line_cmd = Twist()
        self.get_logger().info('MovingAdapter initialized.')

    def _on_line_cmd(self, msg: Twist):
        self._line_cmd = msg
        if self._current_moving_type == 0: 
            self.pub_cmd_vel.publish(self._line_cmd)

    def _on_moving_state(self, msg: MovingParam):
        self._current_moving_type = msg.moving_type
        self._publish_action_cmd()

    def _on_max_vel(self, msg: Float64):
        self._current_max_vel = msg.data
        self._publish_action_cmd()

    def _publish_action_cmd(self):
        if self._current_moving_type == 0:
            return 
            
        cmd = Twist()
        MOVING_STOP = 1
        MOVING_FORWARD = 2
        MOVING_LEFT = 3
        MOVING_RIGHT = 4
        
        if self._current_moving_type == MOVING_STOP:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif self._current_moving_type == MOVING_FORWARD:
            cmd.linear.x = self._current_max_vel
            cmd.angular.z = 0.0
        elif self._current_moving_type == MOVING_LEFT:
            cmd.linear.x = 0.0  
            cmd.angular.z = 0.3 
        elif self._current_moving_type == MOVING_RIGHT:
            cmd.linear.x = 0.0  
            cmd.angular.z = -0.3  
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        self.pub_cmd_vel.publish(cmd)

def main():
    rclpy.init()
    node = MovingAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()