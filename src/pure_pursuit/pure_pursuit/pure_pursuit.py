#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class PurePursuitLogic:
    def __init__(self, wheelbase, waypoints):
        self.L = wheelbase
        self.waypoints = waypoints
        self.num_waypoints = len(waypoints)
        self.current_idx = 0

    #transform from map frame to baselink frame
    def transform_point_to_car_frame(self, car_x, car_y, car_yaw, point):
        """
        Manually transforms a world-frame point to the car's local frame.
        x > 0: In front of the car
        y > 0: To the left of the car
        """
        dx = point[0] - car_x
        dy = point[1] - car_y
        cos_y = np.cos(-car_yaw)
        sin_y = np.sin(-car_yaw)
        # 2D Rotation matrix calculation
        local_x = dx * cos_y - dy * sin_y
        local_y = dx * sin_y + dy * cos_y
        return np.array([local_x, local_y])

    def find_target_waypoint(self, car_x, car_y, car_yaw, lookahead_dist):
        """
        1. Search within a 100-point window from the last found index.
        2. Handle 'loop around' if the window crosses the end of the array.
        3. Enforce forward-half-plane (target must be in front of the car).
        """
        start = self.current_idx
        # Use modulo to create a circular buffer effect
        end = (start + 100) % self.num_waypoints 
        
        final_i = -1
        longest_dist = 0

        # Define the search range based on whether it loops around the array end
        if end < start:
            # Case: Window crosses the finish line (e.g., from index 950 to 50)
            search_range = list(range(start, self.num_waypoints)) + list(range(0, end))
        else:
            # Case: Normal sequential search
            search_range = range(start, end)

        for i in search_range:
            # The ith waypoint x and y
            p_world = self.waypoints[i, :2]
            # Euclidean distance from car to waypoint
            dist = np.linalg.norm(p_world - np.array([car_x, car_y]))
            
            # Transform the world-frame point to the car's local frame
            p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, p_world)
            
            # CRITICAL CONDITIONS:
            # 1. dist <= lookahead_dist: Within the lookahead circle.
            # 2. dist >= longest_dist: Pick the furthest point in the circle for smoothness.
            # 3. p_car[0] > 0: The point MUST be in front of the car (Forward Fix).
            if dist <= lookahead_dist and dist >= longest_dist and p_car[0] > 0:
                longest_dist = dist
                final_i = i

        # Fallback Logic: If no point found in the local window, return the last known index
        if final_i != -1:
            self.current_idx = final_i
        else:
            distances = np.linalg.norm(self.waypoints[:, :2] - np.array([car_x, car_y]), axis=1)
            # Find indices where point is in front
            for i in np.argsort(distances): # Check points starting from the closest
                p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, self.waypoints[i, :2])
                if p_car[0] > 0: # If it's in front, snap to it
                    final_i = i
                    break

        target_pt_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, self.waypoints[final_i, :2])
        return target_pt_car, longest_dist, final_i

    def calculate_steering(self, target_point, lookahead_dist, k_p):
        y = target_point[1]
        # Calculated from https://docs.google.com/presentation/d/1jpnlQ7ysygTPCi8dmyZjooqzxNXWqMgO31ZhcOlKVOE/edit#slide=id.g63d5f5680f_0_33
        safe_la = max(lookahead_dist, 0.1)
        steering_angle = k_p * (2.0 * y) / (safe_la**2)
        if np.isnan(steering_angle) or np.isinf(steering_angle):
            steering_angle = 0.0
        return steering_angle

class ControllerManager(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter("waypoints_path", "/sim_ws/src/pure_pursuit/racelines/arc.csv")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("min_lookahead", 1.0)
        self.declare_parameter("max_lookahead", 2.0)
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
        self.waypoints = np.loadtxt(self.path, delimiter=',', skiprows=1) # Assume x, y, v
        self.planner = PurePursuitLogic(self.wheelbase, self.waypoints)
        self.curr_velocity = 0.0

        # 3. Pubs & Subs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        
        self.get_logger().info("Pure Pursuit Node Started")
        self.viz_pub = self.create_publisher(Marker, '/waypoint_markers', 10)
        self.path_viz_pub = self.create_publisher(Marker, '/full_track_path', 10)
        # Trigger the path visualization once at the start
        # (Wait a tiny bit for RViz to connect)
        self.create_timer(1.0, self.publish_static_path)

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

        # Planning: Use the Logic class to find the target
        target_pt_car, actual_la, target_idx = self.planner.find_target_waypoint(
            car_x, car_y, car_yaw, lookahead_dist
        )

        # Control: Calculate steering and fetch speed from waypoints
        if target_idx != -1:
            self.visualize_lookahead_point(self.waypoints[target_idx])

        if target_idx == -1:
            self.get_logger().warn("Waiting for valid waypoint...")
            self.publish_drive(0.0, 0.0) 
            return
        steer = self.planner.calculate_steering(target_pt_car, actual_la, self.kp)
        # Target velocity is scaled by a safety percentage
        target_vel = self.waypoints[target_idx, 2] * self.vel_percent
        
        # 5. Actuation: Send commands to the car
        self.publish_drive(steer, target_vel)

    def publish_drive(self, steer, vel):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(vel)
        drive_msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(drive_msg)
    
    def visualize_lookahead_point(self, point):
        """
        Publishes a marker to visualize the current lookahead point in RViz.
        :param point: A list or array [x, y] in the 'map' frame.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_point"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set the scale of the sphere (diameter in meters)
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        
        # Set the color (RGBA) - Bright Red for visibility
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        # Set the position of the marker
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0 # Waypoints are on the 2D plane
        
        # Publish the marker
        self.viz_pub.publish(marker)
    def publish_static_path(self):
        """
        Publishes all waypoints from the CSV as a single continuous green line.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "static_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP # This connects all points in order
        marker.action = Marker.ADD
        
        # Line width
        marker.scale.x = 0.1 
        
        # Color: Green (so it contrasts with your red lookahead dot)
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        # Add all waypoints from your loaded CSV to the marker
        for wp in self.waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.0
            marker.points.append(p)
        
        # If it's a loop, connect the last point to the first
        if len(self.waypoints) > 0:
            p_start = Point()
            p_start.x = float(self.waypoints[0][0])
            p_start.y = float(self.waypoints[0][1])
            marker.points.append(p_start)

        self.path_viz_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ControllerManager())
    rclpy.shutdown()

if __name__ == '__main__':
    main()