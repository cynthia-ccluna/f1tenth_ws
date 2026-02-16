import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class GapFinderNode(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        self.subscription = self.create_subscription(LaserScan,lidarscan_topic,self.lidar_callback,10)
        # TODO: Publish to drive
        self.publisher = self.create_publisher(AckermannDriveStamped,drive_topic,10)

        self.safety_bubble_diameter = 0.2
        self.lookahead = 10
        self.speed = 0.8
        self.view_angle = np.pi
        self.disparity_threshold = 0.6
        self.disparity_bubble_diameter = 0.2
        self.fov_bounds = None
        self.initialize = True
        self.do_preprocess = False
        self.do_limit_fov = True
        self.do_mark_minimum = True
        self.do_mark_disparity = True

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc_ranges = ranges.copy()
        proc_ranges[proc_ranges > 3] = self.lookahead
        for i in range(2,len(proc_ranges)-2):
            proc_ranges[i] = (ranges[i-2] + ranges[i-1] + ranges[i] + ranges[i+1] + ranges[i+2])/5
        return proc_ranges

    def update(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        angle_increment = scan_msg.angle_increment

        if self.initialize:
            self.middle_index = ranges.shape[0]//2
            
            #Field of view bounds
            view_angle_count = self.view_angle//angle_increment
            lower_bound = int((ranges.shape[0]-view_angle_count)/2)
            upper_bound = int(lower_bound+view_angle_count)
            self.fov_bounds = [lower_bound, upper_bound+1]
            self.initialize = False

            #center priority mask

        #Limit field of view
        if (self.do_limit_fov):
            limited_ranges = ranges[self.fov_bounds[0]:self.fov_bounds[1]]
        else:
            limited_ranges = ranges

        #Preprocess lidar messages
        if self.do_preprocess:
            proc_ranges = self.preprocess_lidar(limited_ranges)
        else:
            proc_ranges = limited_ranges.copy()
        
        gap_finding_ranges = proc_ranges.copy()
        #Mark large disparity
        mark_indexes = []
        if self.do_mark_disparity:
            for idx in range(1, proc_ranges.shape[0]):
                if abs(proc_ranges[idx]-proc_ranges[idx-1])>self.disparity_threshold:
                    if proc_ranges[idx]<proc_ranges[idx-1]:
                        r_disp = proc_ranges[idx]
                        c_idx = idx
                    else:
                        r_disp = proc_ranges[idx-1]
                        c_idx = idx-1
                    arc = angle_increment * r_disp
                    if arc>0:
                        radius_count =  int((self.disparity_bubble_diameter/2)/arc)
                    else:
                        radius_count = 0
                    l_bound = max(0,c_idx-radius_count)
                    u_bound = min(c_idx+radius_count+1, len(proc_ranges))
                    mark_indexes.append([l_bound,u_bound,r_disp])

        #Find the nearest point
        r = np.min(proc_ranges)
        i = np.argmin(proc_ranges)

        #Mark all points inside 'bubble' 
        arc = angle_increment * r
        if arc>0:
            radius_count =  int((self.safety_bubble_diameter/2)/arc)
        else:
            radius_count = 0
        l_bound = max(0,i-radius_count)
        u_bound = min(i+radius_count+1, len(proc_ranges))
        mark_indexes.append([l_bound,u_bound,r])

        #Apply safety bubble to marked indexes
        for mark_point in mark_indexes:
            proc_ranges[mark_point[0]:mark_point[1]] = 0.0

        # gap_map = proc_ranges.copy()
        # for l_b, u_b, _ in mark_indexes:
        #     gap_map[l_b:u_b] = 0           

        #Find the deepest gap
        max_gap_index = np.argmax(proc_ranges)
        steering = angle_increment * (max_gap_index-proc_ranges.shape[0]//2)

        # steering = 0.3
        print(f"Steering: {steering}")
        ackermann = {"speed": self.speed, "steering": steering}

        return ackermann

    def lidar_callback(self, scan_msg):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        drive = self.update(scan_msg)
        drive_msg = AckermannDriveStamped()

        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "drive"

        #Clipping steering
        steering = drive["steering"]
        steering = max(min(steering, 0.4189), -0.4189)

        drive_msg.drive.speed = float(drive["speed"])
        drive_msg.drive.steering_angle = float(steering)
        self.publisher.publish(drive_msg)

        


def main(args=None):
    rclpy.init(args=args)
    print("GapFollow Initialized")
    gapfinder = GapFinderNode()
    rclpy.spin(gapfinder)
    gapfinder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()