#!/usr/bin/env python
import os
import tf
import rospy
import time
import datetime

from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt


ty_list = []
time_list = []

def odom_postition_callback(msg):
    # Camera position in map frame
    ty = msg.pose.pose.position.y # y -> green axis
    ty_list.append(ty)
    current_time = rospy.Time.now().to_sec()
    time_list.append(current_time)
    rospy.loginfo(f"Current y is {ty}")

def draw_ty_plot(ty_list, time_list):
    if not ty_list or not time_list:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    # Scatter plot of y positions over time
    plt.scatter(time_list, ty_list, color='blue', label='y positions', alpha=0.6)
    plt.plot(time_list, ty_list, color='cyan', linestyle='--', alpha=0.5)

    # Labels and title
    plt.title('Distribution of Y Positions Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Y Position')
    plt.grid(alpha=0.3)
    plt.legend()

    # Generate a filename with the current date and time (up to the minute)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')  # Format: YYYYMMDD_HHMM
    image_filename = f"vis_data/ty_distribution/ty_distribution_{current_time}.png"
    data_filename = f"vis_data/ty_distribution/ty_distribution_{current_time}.txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(image_filename), exist_ok=True)

    # Save the plot as an image
    plt.savefig(image_filename)
    plt.show()
    print(f"Plot saved as {image_filename}")

    # Save time_list and error_list to a .txt file
    with open(data_filename, 'w') as file:
        file.write("Time (s), Y\n")  # Write a header
        for t, e in zip(time_list, ty_list):
            file.write(f"{t}, {e}\n")
    
    print(f"Data saved as {data_filename}")  

def main():
    rospy.init_node('plot_ty')
    # Create subscribers
    rospy.Subscriber("/zed2i/zed_node/odom", Odometry, odom_postition_callback)

    # Spin to keep the script running
    rospy.spin()

    draw_ty_plot(ty_list, time_list)

if __name__ == '__main__':
    main()
