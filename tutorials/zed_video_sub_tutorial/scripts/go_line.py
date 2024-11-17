#!/usr/bin/env python
import rospy
import time
import datetime
import tf
from std_msgs.msg import Int8
from mmdet.serial_util.SerialCommunicate import serial_init
from mmdet.serial_util.control_car import send_command
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

# Car Control Parameters
# --------------------------------
# Panel angle
panel_angel_prev = 0
panel_angle = 0
# Car speed 
speed = 200
left_speed = 200
right_speed = 200
# --------------------------------
yaw_info = 0
# PID Parameters
# --------------------------------
# PID参数
Kp = 200.0  # 比例增益
Ki = 0.0  # 积分增益
Kd = 0.0  # 微分增益
# 设定目标转向角（例如0度代表沿y轴正方向）
target_angle = 0.0  # 目标角度
# PID误差积累
integral = 0.0
previous_error = 0.0
# 用于记录误差和时间
error_list = []
time_list = []
# --------------------------------

def odom_yaw_callback(msg):
    global yaw_info
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
    yaw_info = yaw

    # rospy.loginfo("Current Yaw: %f", yaw)
    adjust_left_right_speed(yaw)

def pid_control(current_angle):
    global integral, previous_error, error_list, time_list
    # 计算当前时间
    current_time = rospy.Time.now().to_sec()
    # 计算角度误差
    error = target_angle - current_angle
    # 将角度差限制在[-180, 180]之间
    if error > 180:
        error -= 360
    elif error < -180:
        error += 360
    # 积分项
    integral += error
    
    # 微分项
    derivative = error - previous_error
    previous_error = error
    
    # 计算调整值
    correction = Kp * error + Ki * integral + Kd * derivative
    # Debug logs
    # rospy.loginfo(f"Error: {error}, Integral: {integral}, Derivative: {derivative}, Correction: {correction}")    
    # 记录误差和时间
    error_list.append(error)
    time_list.append(current_time)
    return correction

def adjust_left_right_speed(current_angle):
    global left_speed, right_speed, speed
    # 根据转向角调整左右轮速度
    correction = pid_control(current_angle)
    left_speed = speed - correction
    right_speed = speed + correction

def draw_error_plot():
    # At program exit, plot and save the error curve
    plt.figure()
    plt.plot(time_list, error_list, label="Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title("PID Error Over Time")
    plt.legend()
    plt.grid()

    # Generate a filename with the current date and time (up to the minute)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')  # Format: YYYYMMDD_HHMM
    filename = f"vis_data/error_plot/pid_error_plot_{current_time}.png"

    # Save the plot as an image
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved as {filename}")

def main():
    rospy.init_node('control_car', anonymous=True)

    serial_init()  # Initialize and open serial port
    time.sleep(3)  # Wait for 1 seconds
    send_command(0,0,0,0)
    time.sleep(1)  # Wait for 1 seconds
    rospy.Subscriber("/zed2i/zed_node/odom", Odometry, odom_yaw_callback)
    # Define a rate to control how often the loop runs
    rate = rospy.Rate(120)  # 120 Hz

    while not rospy.is_shutdown():
        # Send the current panel angle to the car's control system
        # rospy.loginfo(f"panel_angle is {panel_angle}")
        rospy.loginfo(f"left_speed is {left_speed}, right_speed is {right_speed}")
        rospy.loginfo(f"yaw is {yaw_info}")
        # if(panel_angle == 90 and panel_angel_prev == 30):
        #     time.sleep(0.1)
        # send_command(panel_angle, left_speed, right_speed, speed)
        send_command(panel_angle, left_speed, right_speed, speed)
        time.sleep(1)

        # Sleep to maintain the loop rate
        rate.sleep()

    send_command(0, 0, 0, 0)
    time.sleep(1) 

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:       
        # pass
        draw_error_plot()