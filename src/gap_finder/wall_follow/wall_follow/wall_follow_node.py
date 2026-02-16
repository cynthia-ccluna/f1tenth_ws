import math
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')
        self.get_logger().info("!!!!!!!! 這是 2026 版本 !!!!!!!!")

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.subscription = self.create_subscription(
            LaserScan, 
            lidarscan_topic, 
            self.scan_callback, 
            10)
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped, 
            drive_topic, 
            10)
        
        # TODO: set PID gains
        self.kp = 1
        self.kd = 0
        self.ki = 0.01

        # TODO: store history
        self.integral = 0
        self.prev_error = 0
        self.error = 0

        # TODO: store any necessary values you think you'll need
        self.lookahead = 0.5
        self.desired_dist = 1

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """

        #TODO: implement
        angle_increment = (range_data.angle_max-range_data.angle_min)/len(range_data.ranges)
        index = int((angle-range_data.angle_min)/angle_increment)
        index = max(0, min(index, len(range_data.ranges)-1))
        range_val = range_data.ranges[index]
        if math.isnan(range_val) or math.isinf(range_val):
            return 10
        return range_val

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        #TODO:implement
        angle_a = math.radians(45)
        angle_b = math.radians(90)
        theta = angle_b-angle_a
        a = self.get_range(range_data, angle_a)
        b = self.get_range(range_data, angle_b)
        alpha = math.atan2((a*math.cos(theta)-b),(a*math.sin(theta)))
        d_curr = b*math.cos(alpha)
        d_pred = d_curr+self.lookahead*math.sin(alpha)
        error = dist-d_pred
        return error

    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        angle = 0.0
        # TODO: Use kp, ki & kd to implement a PID controller
        self.integral += error
        derivative = error-self.prev_error
        steering_angle = self.kp*error + self.kd*derivative + self.ki*self.integral
        self.prev_error = error
        steering_angle = max(-0.4, min(steering_angle, 0.4))
        drive_msg = AckermannDriveStamped()
        # TODO: fill in drive message and publish
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = steering_angle
        self.publisher_.publish(drive_msg)

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        print(">>>>>> CALLBACK TRIGGERED <<<<<<")
        try:
            error = self.get_error(msg, self.desired_dist) # TODO: replace with error calculated by get_error()
            # TODO: calculate desired car velocity based on error
            velocity = 1.5 if abs(error) < 0.2 else 0.5 
            self.pid_control(error, velocity) # TODO: actuate the car with PID
        except Exception as e:
            # 如果出錯，這裡會印出紅色的報錯訊息，讓我們知道為什麼沒 publisher
            self.get_logger().error(f"Callback 崩潰了！原因: {e}")


# def main(args=None):
#     rclpy.init(args=args)
#     print("WallFollow Initialized")
#     wall_follow_node = WallFollow()
#     rclpy.spin(wall_follow_node)

#     # Destroy the node explicitly
#     # (optional - otherwise it will be done automatically
#     # when the garbage collector destroys the node object)
#     wall_follow_node.destroy_node()
#     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    # 使用 print 並加上 flush=True，保證訊息不會被存在緩衝區
    print(">>> 節點嘗試啟動中...", flush=True)
    try:
        wall_follow_node = WallFollow()
        print(">>> 節點物件建立成功！", flush=True)
        rclpy.spin(wall_follow_node)
    except Exception as e:
        print(f">>> 崩潰原因: {e}", flush=True)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()