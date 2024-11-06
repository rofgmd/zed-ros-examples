#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import mmcv
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import load_config, get_input_shape
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

# Initialize CvBridge for converting ROS images to OpenCV format
bridge = CvBridge()

# Load your model and configuration once
deploy_cfg = '/home/kevin/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py'
model_cfg = '/home/kevin/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'
device = 'cuda'
backend_model = ['/home/kevin/mmdeploy_model/rtmdet-ins/end2end.engine']

output_image_pub = rospy.Publisher('/segmentation_result', Image, queue_size=60)

def init_model():
    global task_processor, model, input_shape
    deploy_cfg_loaded, model_cfg_loaded = load_config(deploy_cfg, model_cfg)
    task_processor = build_task_processor(model_cfg_loaded, deploy_cfg_loaded, device)
    model = task_processor.build_backend_model(backend_model)
    input_shape = get_input_shape(deploy_cfg)

# Inference function that takes image data in NumPy format
def inference_image(image_np):
    model_inputs, _ = task_processor.create_input(image_np, input_shape)
    with torch.no_grad():
        result = model.test_step(model_inputs)
    return result[0]

def display_result(image_np, result):
    img = mmcv.imconvert(image_np, 'bgr', 'rgb')
    visualizer = task_processor.get_visualizer('visualize','')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.3,
        show=False)

    img = visualizer.get_image()
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    segment_result = bridge.cv2_to_imgmsg(img, "rgb8")
    output_image_pub.publish(segment_result)
    cv2.imshow('result', img)
    cv2.waitKey(1)    

def publish_result(image_np, result):
    img = mmcv.imconvert(image_np, 'bgr', 'rgb')
    visualizer = task_processor.get_visualizer('visualize','')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.3,
        show=False)
    img = visualizer.get_image()
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    segment_result = bridge.cv2_to_imgmsg(img, "rgb8")
    output_image_pub.publish(segment_result)

def publish_mask(result):
    print(result)

# Callback function to process each received image
def image_callback(ros_image):
    # Convert ROS image to OpenCV format (BGR)
    image_np = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    # Run inference
    start_time = time.time()
    result = inference_image(image_np)
    process_time = time.time() - start_time
    rospy.loginfo(f"Inference time: {process_time:.4f}s")

    # Process result (display, save, or further processing)
    # Here, you could convert the segmentation masks to an output format of your choice
    
    # Example: print segmentation result
    # print(result)  # Placeholder for your actual result processing

    # publish result in ros topic
    publish_result(image_np, result)
    publish_mask(result)

    # Display result in real time
    # display_result(image_np, result)

def main():
    # Initialize ROS node
    rospy.init_node('zed_tensorRT_segmenter', anonymous=True)

    # Load the TensorRT model
    init_model()

    # Subscribe to the ZED camera image topic
    rospy.Subscriber("/zed2i/zed_node/rgb_raw/image_raw_color", Image, image_callback)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()