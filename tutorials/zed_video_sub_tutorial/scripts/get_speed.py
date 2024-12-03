import pyzed.sl as sl
import cv2
import numpy as np
import time

# 初始化 ZED 2i 相机
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # 设置分辨率
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # 设置深度模式
init_params.coordinate_units = sl.UNIT.METER  # 使用米作为单位

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("无法打开 ZED 2i 相机")
    exit()

# 初始化相关变量
runtime_params = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()

orb = cv2.ORB_create()  # 初始化 ORB 特征提取器

# 获取第一帧作为参考帧
zed.grab(runtime_params)
zed.retrieve_image(image, sl.VIEW.LEFT)
zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

prev_frame = image.get_data()  # RGB 图像
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

fps = 30  # 假设帧率为 30 FPS，可根据实际调整
prev_time = time.time()
speed_list = []  # 用于存储平滑的速度值

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # 获取当前帧图像和深度图
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        frame = image.get_data()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测当前帧的特征点
        kp, des = orb.detectAndCompute(gray_frame, None)

        # 匹配特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # 匹配点位移计算
        total_displacement = 0
        valid_matches = 0
        displacement_threshold = 10  # 最大像素位移阈值

        for match in matches:
            prev_pt = prev_kp[match.queryIdx].pt
            curr_pt = kp[match.trainIdx].pt
            displacement = np.linalg.norm(np.array(curr_pt) - np.array(prev_pt))

            prev_depth = depth.get_value(int(prev_pt[0]), int(prev_pt[1]))[1]
            curr_depth = depth.get_value(int(curr_pt[0]), int(curr_pt[1]))[1]

            # 过滤深度值和位移异常点
            if 0.1 < prev_depth < 20 and 0.1 < curr_depth < 20 and displacement < displacement_threshold:
                real_displacement = displacement * (curr_depth + prev_depth) / 2
                total_displacement += real_displacement
                valid_matches += 1

        # 计算平均位移和速度
        if valid_matches > 0:
            avg_displacement = total_displacement / valid_matches

            # 动态计算帧间时间差
            curr_time = time.time()
            delta_time = curr_time - prev_time
            if delta_time > 0:
                speed = avg_displacement / delta_time

                # 平滑速度值
                speed_list.append(speed)
                if len(speed_list) > 5:
                    speed_list.pop(0)
                smoothed_speed = sum(speed_list) / len(speed_list)
                print(f"Smoothed Speed: {smoothed_speed:.2f} m/s")

            prev_time = curr_time

        # 显示匹配结果
        matched_frame = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches[:10], None, flags=2)
        cv2.imshow("Feature Matching with ZED 2i", matched_frame)

        # 更新前一帧数据
        prev_frame = frame
        prev_gray = gray_frame
        prev_kp, prev_des = kp, des

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭相机和窗口
zed.close()
cv2.destroyAllWindows()
