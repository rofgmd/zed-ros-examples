#!/usr/bin/env python
import time
from mmdet.serial_util.SerialCommunicate import serial_init
from mmdet.serial_util.control_car import send_command

panel_angle = 30

def main():
    serial_init()
    time.sleep(1)
    send_command(panel_angle, 0, 0, 0)
    time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")