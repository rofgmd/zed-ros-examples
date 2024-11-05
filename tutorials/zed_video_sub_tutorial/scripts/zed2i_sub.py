#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

# Define the callback functions for each subscriber
def image_right_rectified_callback(data):
    rospy.loginfo("Received right rectified image data")
    # Process the right rectified image data here

def image_left_rectified_callback(data):
    rospy.loginfo("Received left rectified image data")
    # Process the left rectified image data here

def image_rectified_callback(data):
    rospy.loginfo(f"Rectified image received from ZED - Size: {data.width}, {data.height}")
    # Process the rectified image data here

def main():
    # Initialize the ROS node
    rospy.init_node('zed_video_subscriber', anonymous=True)

    # Set up subscribers for the topics
    # sub_right_rectified = rospy.Subscriber("/zed2i/zed_node/right/image_rect_color", Image, image_right_rectified_callback)
    # sub_left_rectified = rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, image_left_rectified_callback)
    sub_left_rectified = rospy.Subscriber("/zed2i/zed_node/rgb_raw/image_raw_color", Image, image_rectified_callback)

    # Keep the program alive until manually interrupted
    rospy.spin()

if __name__ == '__main__':
    main()