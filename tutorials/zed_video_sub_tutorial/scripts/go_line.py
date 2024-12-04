#!/usr/bin/env python
import rospy
import time
import datetime
import tf
import os
from std_msgs.msg import Int8
from mmdet.serial_util.SerialCommunicate import serial_init
from mmdet.serial_util.control_car import send_command
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from collections import deque

# 滤波窗口大小
filter_window = 5
angle_buffer = deque(maxlen=filter_window)
ty_buffer = deque(maxlen=filter_window)

# Car Control Parameters
# --------------------------------
# Panel angle
panel_angel_prev = 0
panel_angle = 0
# Car speed 
speed = 100
left_speed = 100
right_speed = 100
# --------------------------------
ty_info = 0
# PID Parameters
# --------------------------------
# PID参数
# Kp = 125.0  # 比例增益
# Ki = 0.045  # 积分增益
# Kd = 0.7  # 微分增益
# --------------------------------
Kp_ty = 125
Ki_ty = 0.045
Kd_ty = 0.7
# 设定目标转向角（例如0度代表沿y轴正方向）
# target_angle = 0.0  # 目标角度
target_ty = 0.0
# PID误差积累
# integral = 0.0
integral_ty = 0.0
# previous_error = 0.0
previous_error_ty = 0.0
# 用于记录误差和时间
error_list = []
error_list_ty = []
time_list = []
time_list_ty = []
# --------------------------------

def smooth_data(data, buffer):
    buffer.append(data)
    return sum(buffer) / len(buffer)

def odom_y_callback(msg):
    global ty_info,ty_buffer
    ty = msg.pose.pose.position.y # y -> green axis
    # 平滑位置数据
    ty_info = smooth_data(ty, ty_buffer)
    adjust_left_right_speed(ty_info)   

def pid_control_ty(current_ty):
    global previous_error_ty
    # 计算当前时间
    current_time = rospy.Time.now().to_sec()

    # 计算位置误差
    error = target_ty - current_ty
    # 积分项
    global integral_ty
    integral_ty += error
    
    # 微分项
    derivative = error - previous_error_ty
    previous_error_ty = error
    
    # 计算调整值
    correction = Kp_ty * error + Ki_ty * integral_ty + Kd_ty * derivative
   
    # 记录误差和时间
    error_list_ty.append(error)
    time_list_ty.append(current_time)
    return correction

def adjust_left_right_speed(ty):
    global left_speed, right_speed
    correction = pid_control_ty(ty)
    # 根据误差调整左右轮速度
    left_speed = speed - correction
    right_speed = speed + correction

def draw_error_plot():
    # At program exit, plot and save the error curve
    rospy.loginfo("Draw Error plot")
    plt.figure()
    plt.plot(time_list_ty, error_list_ty, label="Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title("PID Error Over Time")
    plt.legend()
    plt.grid()

    # Generate a filename with the current date and time (up to the minute)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')  # Format: YYYYMMDD_HHMM
    image_filename = f"vis_data/error_plot/pid_error_plot_{current_time}.png"
    data_filename = f"vis_data/error_plot/pid_error_data_{current_time}.txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(image_filename), exist_ok=True)

    # Save the plot as an image
    plt.savefig(image_filename)
    plt.show()
    print(f"Plot saved as {image_filename}")

    # Save time_list and error_list to a .txt file
    with open(data_filename, 'w') as file:
        file.write("Time (s), Error\n")  # Write a header
        for t, e in zip(time_list, error_list):
            file.write(f"{t}, {e}\n")
    
    print(f"Data saved as {data_filename}")

# Function to draw the x distribution plot
def draw_y_distribution_plot():
    if not time_list_ty or not error_list_ty:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(10, 6))

def main():
    rospy.init_node('control_car', anonymous=True)

    serial_init()  # Initialize and open serial port
    time.sleep(1)  # Wait for 1 seconds
    rospy.Subscriber("/zed2i/zed_node/odom", Odometry, odom_y_callback)
    # Define a rate to control how often the loop runs
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        # Send the current panel angle to the car's control system
        # rospy.loginfo(f"panel_angle is {panel_angle}")
        rospy.loginfo(f"left_speed is {left_speed}, right_speed is {right_speed}")
        rospy.loginfo(f"ty is {ty_info}")
        send_command(panel_angle, left_speed, right_speed, speed)

        # Sleep to maintain the loop rate
        rate.sleep()

    send_command(0, 0, 0, 0)
    time.sleep(1) 
    draw_error_plot()
    draw_y_distribution_plot()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:       
        pass