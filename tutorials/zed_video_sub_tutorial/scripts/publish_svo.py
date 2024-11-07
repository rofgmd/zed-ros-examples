#!/usr/bin/env python
import rospy
import cv2
import pyzed.sl as sl
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image

def cam_init():
    # 创建ZED相机对象
    cam = sl.Camera()

    # 初始化
    init_params = sl.InitParameters()
    init_params.set_from_svo_file("/home/kevin/Documents/test_ros/data/1.svo")  # 设置SVO文件路径
    # init_params.svo_real_time_mode = False  # 非实时模式，SVO文件按自己的速度处理
    init_params.svo_real_time_mode = True  # 实时模式
    # 打开相机
    err = cam.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error: {err}")
        exit(1)    
    
    return cam

def publish_rgbd(cam):
    # Initialize ROS node and publishers
    rospy.init_node('svo_rgbd_publisher', anonymous=True)
    rgb_pub = rospy.Publisher('/zed2i/zed_node/rgb_raw/image_raw_color', Image, queue_size=60)
    depth_pub = rospy.Publisher('/zed2i/zed_node/depth/depth_registered', Image, queue_size=60)
    
    # Create a CvBridge to convert images
    bridge = CvBridge()
    
    # Image containers for RGB and depth
    image = sl.Mat()
    depth_map = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    while not rospy.is_shutdown():
        # Capture images from ZED camera
        if cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left RGB image
            cam.retrieve_image(image, sl.VIEW.LEFT)
            rgb_image = image.get_data()

            # Convert the image from RGBA (8UC4) to RGB (8UC3)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)

            # Retrieve depth map
            cam.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            depth_image = depth_map.get_data()
            
            # Convert images to ROS format and publish
            rgb_msg = bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
            depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
            rgb_pub.publish(rgb_msg)
            depth_pub.publish(depth_msg)
            
            rospy.loginfo("Published RGB and Depth images.")
        else:
            break  # End of SVO file

if __name__ == '__main__':
    cam = cam_init()
    try:
        publish_rgbd(cam)
    except rospy.ROSInterruptException:
        pass
    finally:
        cam.close()