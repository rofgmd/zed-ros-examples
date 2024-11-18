#!/usr/bin/env python

import time
from mmdet.serial_util.control_car import send_command
from mmdet.serial_util.SerialCommunicate import serial_init

def main():
    print("Serial Init start")
    serial_init()
    time.sleep(1)

    print("Clean buffer")
    send_command(0,0,0,0)
    time.sleep(1)

if __name__ == "__main__":
    main()