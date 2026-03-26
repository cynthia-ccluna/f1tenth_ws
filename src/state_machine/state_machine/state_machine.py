#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        # Definition of States
        self.GB_TRACK = "GB_TRACK"    
        self.FTGONLY = "FTGONLY"  
        self.current_state = self.GB_TRACK

        self.safety_dist = 2.0 
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.state_pub = self.create_publisher(String, '/state', 10)

        self.get_logger().info("--- State Machine Started: Defaulting to GB_TRACK ---")

    def scan_callback(self, msg):
            num_points = len(msg.ranges)
            mid = num_points // 2
            window = 30  #around 10 degree        
            front_view = msg.ranges[mid - window : mid + window]
            
            # filter out unreasonably small value
            valid_ranges = [r for r in front_view if r > 0.1]
            if not valid_ranges:
                return

            min_dist = min(valid_ranges)

            new_state = self.current_state
            if min_dist < self.safety_dist:
                new_state = self.FTGONLY
            else:
                new_state = self.GB_TRACK

            # print out log at state transition
            if new_state != self.current_state:
                if new_state == self.FTGONLY:
                    self.get_logger().warn(f"[DETECTED] Obstacle at {min_dist:.2f}m! Switching to FTGONLY")
                else:
                    self.get_logger().info(f"[CLEAR] Path is clear. Returning to GB_TRACK")
                
                self.current_state = new_state

            # publish state to controller
            state_msg = String()
            state_msg.data = self.current_state
            self.state_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
