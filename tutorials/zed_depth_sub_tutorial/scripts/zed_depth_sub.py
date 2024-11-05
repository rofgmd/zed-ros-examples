#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image

def depthCallback(msg):
    # Convert the raw data to a NumPy array of floats
    depths = np.frombuffer(msg.data, dtype=np.float32)

    # Calculate the center pixel coordinates
    u = msg.width // 2
    v = msg.height // 2

    # Linear index for the center pixel
    centerIdx = u + msg.width * v

    # Output the center pixel distance
    rospy.loginfo(f"Center distance : {depths[centerIdx]:.2f} m")

def main():
    # Initialize the ROS node
    rospy.init_node('zed_video_depth_subscriber', anonymous=True)

    # Set up subscribers for the topics
    subDepth = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, depthCallback)

    # Keep the program alive until manually interrupted
    rospy.spin()

if __name__ == '__main__':
    main()