import rclpy, cv2, cv_bridge, numpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool 

class Follower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.get_logger().info("Start line follower. 4-Sentinel Strict Crossroad Detection activated.")
        self.bridge = cv_bridge.CvBridge()

        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/line_follow/cmd_vel', 10)
        self.line_found_pub = self.create_publisher(Bool, '/line_follow/line_found', 10)
        self.crossroad_pub = self.create_publisher(Bool, '/line_follow/crossroad', 10)
        self.pub = self.create_publisher(Image, '/camera/process_image', 10)
        
        self.twist = Twist()
        self.base_speed = 0.1       
        self.turn_kp_divider = 400.0 

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_black = numpy.array([0, 0, 0])
        upper_black = numpy.array([180, 255, 180])  
        mask = cv2.inRange(hsv, lower_black, upper_black)

        h, w, d = image.shape

        # ==========================================
        # 1. 严格 4 向哨兵检测 (免疫 T 型、Y 型和斜线)
        # ==========================================
        # 划分四个严格的检测区域
        top_zone = mask[0 : int(h*0.35), int(w*0.4) : int(w*0.6)]         # 正上方
        bot_zone = mask[int(h*0.65) : h, int(w*0.4) : int(w*0.6)]         # 正下方
        left_zone = mask[int(h*0.35) : int(h*0.65), 0 : int(w*0.25)]      # 极左侧
        right_zone = mask[int(h*0.35) : int(h*0.65), int(w*0.75) : w]     # 极右侧

        # 提高像素门槛！必须有大块实心黑线才认，彻底过滤噪点和边缘擦边
        px_threshold = 1000 
        is_crossroad = (cv2.countNonZero(top_zone) > px_threshold) and \
                       (cv2.countNonZero(bot_zone) > px_threshold) and \
                       (cv2.countNonZero(left_zone) > px_threshold) and \
                       (cv2.countNonZero(right_zone) > px_threshold)

        if is_crossroad:
            cv2.putText(image, "CROSSROAD!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 为了让你直观看到这4个哨兵，我在画面上画出蓝色虚线框！
        cv2.rectangle(image, (int(w*0.4), 0), (int(w*0.6), int(h*0.35)), (255, 0, 0), 2)
        cv2.rectangle(image, (int(w*0.4), int(h*0.65)), (int(w*0.6), h), (255, 0, 0), 2)
        cv2.rectangle(image, (0, int(h*0.35)), (int(w*0.25), int(h*0.65)), (255, 0, 0), 2)
        cv2.rectangle(image, (int(w*0.75), int(h*0.35)), (w, int(h*0.65)), (255, 0, 0), 2)

        # ==========================================
        # 2. 巡线控制 (60像素防反光缝隙)
        # ==========================================
        search_top_line = int(h/2) + 10 
        search_bot_line = int(h/2) + 70 
        
        line_mask = mask.copy()
        line_mask[0:search_top_line, 0:w] = 0
        line_mask[search_bot_line:h, 0:w] = 0

        cv2.line(image, (0, search_top_line), (w, search_top_line), (0, 255, 0), 2)
        cv2.line(image, (0, search_bot_line), (w, search_bot_line), (0, 255, 0), 2)

        M = cv2.moments(line_mask)
        line_found_msg = Bool()

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 20, (0,0,255), -1)

            err = cx - w/2
            
            self.twist.linear.x = self.base_speed
            self.twist.angular.z = -float(err) / self.turn_kp_divider
            line_found_msg.data = True
        else:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            line_found_msg.data = False

        self.cmd_vel_pub.publish(self.twist)
        self.line_found_pub.publish(line_found_msg)
        
        cross_msg = Bool()
        cross_msg.data = is_crossroad
        self.crossroad_pub.publish(cross_msg)
            
        self.pub.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)    
    follower = Follower()
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()