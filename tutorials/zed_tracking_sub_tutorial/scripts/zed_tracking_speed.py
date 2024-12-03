#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from datetime import datetime

# Global variables
pre_tx = 0.0
pre_timestamp = 0.0

def odom_speed_callback(msg):
    global pre_tx, pre_timestamp
    
    # Current position and timestamp
    tx = msg.pose.pose.position.x
    timestamp = datetime.now().timestamp()
    
    # Calculate speed (if this is not the first callback)
    if pre_timestamp != 0:
        delta_tx = tx - pre_tx
        delta_time = timestamp - pre_timestamp
        if delta_time > 0:
            speed = delta_tx / delta_time
            rospy.loginfo(f"Current speed: {speed:.3f} m/s")
        else:
            rospy.logwarn("Delta time is zero, skipping speed calculation.")
    
    # Update previous values
    pre_tx = tx
    pre_timestamp = timestamp

def main():
    # Initialize the ROS node
    rospy.init_node('zed_tracking_subscriber')

    # Create subscriber
    rospy.Subscriber("/zed2i/zed_node/odom", Odometry, odom_speed_callback)

    # Spin to keep the script running
    rospy.spin()

if __name__ == '__main__':
    main()
