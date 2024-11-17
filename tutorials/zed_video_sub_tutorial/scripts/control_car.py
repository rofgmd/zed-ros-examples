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
speed = 100
left_speed = 100
right_speed = 100
# --------------------------------

# PID Parameters
# --------------------------------
# PID参数
Kp = 1.0  # 比例增益
Ki = 0.0  # 积分增益
Kd = 0.1  # 微分增益
# 设定目标转向角（例如0度代表沿y轴正方向）
target_angle = 0.0  # 目标角度
# PID误差积累
integral = 0.0
previous_error = 0.0
# --------------------------------

def control_angle_panel_callback(angle_msg):
    global panel_angle, panel_angel_prev
    panel_angel_prev = panel_angle
    panel_angle = angle_msg.data

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

    rospy.loginfo("Current Yaw: %f", yaw)

def pid_control(current_angle):
    global integral, previous_error
    # 计算角度误差
    error = target_angle - current_angle
    # 将角度差限制在[-180, 180]之间
    if error > 180:
        error -= 360
    elif error < -180:
        error += 360

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
        # send_command(panel_angle, left_speed, right_speed, speed)
        send_command(0, left_speed, right_speed, speed)
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