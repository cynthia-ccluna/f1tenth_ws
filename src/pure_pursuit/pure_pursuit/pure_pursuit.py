#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

class PurePursuitLogic:
    def __init__(self, wheelbase):
        self.L = wheelbase

    #transform from map frame to baselink frame
    def transform_waypoints(self, car_x, car_y, car_yaw, waypoints): 
        # Translate
        shifted_x = waypoints[:, 0] - car_x
        shifted_y = waypoints[:, 1] - car_y
        # Rotate (Counter-rotate by car_yaw to get car frame)
        cos_y = np.cos(-car_yaw)
        sin_y = np.sin(-car_yaw)
        local_x = shifted_x * cos_y - shifted_y * sin_y
        local_y = shifted_x * sin_y + shifted_y * cos_y
        return np.column_stack((local_x, local_y))

    def calculate_steering(self, target_point, lookahead_dist, k_p):
        y = target_point[1]
        # Calculated from https://docs.google.com/presentation/d/1jpnlQ7ysygTPCi8dmyZjooqzxNXWqMgO31ZhcOlKVOE/edit#slide=id.g63d5f5680f_0_33
        steering_angle = k_p * (2.0 * y) / (lookahead_dist**2)
        return steering_angle

class ControllerManager(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter("waypoints_path", "/sim_ws/src/pure_pursuit/racelines/traj_race_cl-oct31_v5.csv")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("min_lookahead", 0.5)
        self.declare_parameter("max_lookahead", 1.0)
        self.declare_parameter("lookahead_ratio", 8.0)
        self.declare_parameter("K_p", 0.5)
        self.declare_parameter("steering_limit", 25.0) # Degrees
        self.declare_parameter("velocity_percentage", 0.6)
        self.declare_parameter("wheelbase", 0.33)

        self.path = self.get_parameter("waypoints_path").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value
        self.min_la = self.get_parameter("min_lookahead").value
        self.max_la = self.get_parameter("max_lookahead").value
        self.la_ratio = self.get_parameter("lookahead_ratio").value
        self.kp = self.get_parameter("K_p").value
        self.steer_limit = np.radians(self.get_parameter("steering_limit").value)
        self.vel_percent = self.get_parameter("velocity_percentage").value
        self.wheelbase = self.get_parameter("wheelbase").value

        # 2. Initialize Logic & Data
        self.planner = PurePursuitLogic(self.wheelbase)
        self.waypoints = np.loadtxt(self.path, delimiter=',', skiprows=1) # Assume x, y, v
        self.curr_velocity = 0.0

        # 3. Pubs & Subs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        
        self.get_logger().info("Pure Pursuit Node Started")

    def get_yaw_from_quat(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        # Perception: Get car position
        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)
        self.curr_velocity = msg.twist.twist.linear.x

        # Dynamic Lookahead calculation (change lookahead based on the current velocity)
        lookahead_dist = np.clip(self.max_la * self.curr_velocity / self.la_ratio, self.min_la, self.max_la)

        # Transform all waypoints to car frame
        local_wpts = self.planner.transform_waypoints(car_x, car_y, car_yaw, self.waypoints[:, :2])

        # Find target: First point in front of car that is far enough
        dists = np.linalg.norm(local_wpts, axis=1) # count how far it is from the car to the point
        mask = (local_wpts[:, 0] > 0) & (dists >= lookahead_dist)
        valid_idxs = np.where(mask)[0]
        
        if len(valid_idxs) > 0:
            target_pt = local_wpts[valid_idxs[0]]
            actual_la = dists[valid_idxs[0]]
            
            # Control
            steer = self.planner.calculate_steering(target_pt, actual_la, self.kp)
            steer = np.clip(steer, -self.steer_limit, self.steer_limit)
            
            # Speed from waypoint 
            target_vel = self.waypoints[valid_idxs[0], 2] * self.vel_percent
            
            # Actuation
            self.publish_drive(steer, target_vel)

    def publish_drive(self, steer, vel):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(vel)
        drive_msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ControllerManager())
    rclpy.shutdown()

if __name__ == '__main__':
    main()