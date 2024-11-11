#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf
import math

RAD2DEG = 57.295779513

def odom_callback(msg):
    # Camera position in map frame
    tx = msg.pose.pose.position.x # x -> red axis
    ty = msg.pose.pose.position.y # y -> green axis
    tz = msg.pose.pose.position.z # z -> blue axis

    # Orientation quaternion
    q = msg.pose.pose.orientation
    quaternion = (q.x, q.y, q.z, q.w)

    # 3x3 Rotation matrix from quaternion
    euler = tf.transformations.euler_from_quaternion(quaternion)
    roll, pitch, yaw = euler

    # Output the measure
    rospy.loginfo("Received odom in '%s' frame : X: %.2f Y: %.2f Z: %.2f - R: %.2f P: %.2f Y: %.2f",
                  msg.header.frame_id, tx, ty, tz, roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG)

def pose_callback(msg):
    # Camera position in map frame
    tx = msg.pose.position.x
    ty = msg.pose.position.y
    tz = msg.pose.position.z

    # Orientation quaternion
    q = msg.pose.orientation
    quaternion = (q.x, q.y, q.z, q.w)

    # 3x3 Rotation matrix from quaternion
    euler = tf.transformations.euler_from_quaternion(quaternion)
    roll, pitch, yaw = euler

    # Output the measure
    rospy.loginfo("Received pose in '%s' frame : X: %.2f Y: %.2f Z: %.2f - R: %.2f P: %.2f Y: %.2f",
                  msg.header.frame_id, tx, ty, tz, roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG)

def main():
    # Initialize the ROS node
    rospy.init_node('zed_tracking_subscriber')

    # Create subscribers
    rospy.Subscriber("/zed2i/zed_node/odom", Odometry, odom_callback)
    # rospy.Subscriber("/zed2i/zed_node/pose", PoseStamped, pose_callback)

    # Spin to keep the script running
    rospy.spin()

if __name__ == '__main__':
    main()
