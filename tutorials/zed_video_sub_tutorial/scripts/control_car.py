#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Int8
from mmdet.serial_util.SerialCommunicate import serial_init
from mmdet.serial_util.control_car import send_command

panel_angel_prev = 0
panel_angle = 90

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
        # Send the current panel angle to the car's control system
        rospy.loginfo(f"panel_angle is {panel_angle}")
        # if(panel_angle == 90 and panel_angel_prev == 30):
        #     time.sleep(0.01)
        send_command(panel_angle, 100, 100, 100)
        time.sleep(0.01)

        # Sleep to maintain the loop rate
        rate.sleep()

    send_command(0, 0, 0, 0)
    time.sleep(0.5) 

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:       
        pass