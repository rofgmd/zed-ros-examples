#!/usr/bin/env python
import rospy
import time
import tf
from std_msgs.msg import Int8
from mmdet.serial_util.SerialCommunicate import serial_init
from mmdet.serial_util.control_car import send_command
from nav_msgs.msg import Odometry

# Car Control Parameters
# --------------------------------
# Panel angle
panel_angel_prev = 0
panel_angle = 90
# Car speed 
speed = 0
left_speed = 0
right_speed = 0
# --------------------------------

def control_angle_panel_callback(angle_msg):
    global panel_angle, panel_angel_prev
    panel_angel_prev = panel_angle
    panel_angle = angle_msg.data

def main():
    rospy.init_node('control_car', anonymous=True)

    serial_init()  # Initialize and open serial port
    time.sleep(1)  # Wait for 1 seconds

    rospy.Subscriber('panel_angle', Int8, control_angle_panel_callback)
    # Define a rate to control how often the loop runs

    rate = rospy.Rate(120)  # 120 Hz

    while not rospy.is_shutdown():
        # maintain_time = 0.5
        # Send the current panel angle to the car's control system
        # if panel_angle == 30:
        #     rospy.loginfo(f"Panel angle is 30, maintaining for {maintain_time} second...")
        #     send_command(30, left_speed, right_speed, speed)
        #     time.sleep(maintain_time)
        # else: 
        rospy.loginfo(f"Sending panel angle: {panel_angle}")
        send_command(panel_angle, left_speed, right_speed, speed)

        # Sleep to maintain the loop rate
        rate.sleep()

    send_command(0, 0, 0, 0)
    time.sleep(0.5) 

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:       
        pass