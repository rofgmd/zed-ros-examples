#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import mmcv
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray
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
model_cfg = '/home/kevin/mmdetection/configs/cow/rtmdet-ins_tiny_1xb4-500e_cow_update_20241017.py'
device = 'cuda'
backend_model = ['/home/kevin/mmdeploy/mmdeploy_models/cow/rtmdet-ins_tiny/end2end.engine']

output_image_pub = rospy.Publisher('/segmentation_result/image', Image, queue_size=60)
output_bboxes_pub = rospy.Publisher('/segmentation_result/bboxes', Float32MultiArray, queue_size=60)
output_labels_pub = rospy.Publisher('/segmentation_result/labels', Int32MultiArray, queue_size=60)
output_scores_pub = rospy.Publisher('/segmentation_result/scores', Float32MultiArray, queue_size=60)
output_masks_pub = rospy.Publisher('/segmentation_result/masks', Image, queue_size=60)  # Publish masks as images

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

    # Convert image and publish
    segment_result = bridge.cv2_to_imgmsg(img, "rgb8")
    output_image_pub.publish(segment_result)
    # Publish bounding boxes
    bboxes_msg = Float32MultiArray(data=result.pred_instances.bboxes.cpu().numpy().flatten())
    output_bboxes_pub.publish(bboxes_msg)
    # Publish labels
    labels_msg = Int32MultiArray(data=result.pred_instances.labels.cpu().numpy())
    output_labels_pub.publish(labels_msg)
    # Publish scores
    scores_msg = Float32MultiArray(data=result.pred_instances.scores.cpu().numpy())
    output_scores_pub.publish(scores_msg)

def publish_mask(result):
    pred_instances = result.pred_instances
    masks = pred_instances.masks.cpu().numpy().astype('uint8')  # Convert boolean masks to uint8

    # Combine masks by summing them along the instance axis
    combined_mask = np.sum(masks, axis=0) * 255  # Scale to 0-255

    # Ensure values do not exceed 255 in case of overlapping masks
    combined_mask = np.clip(combined_mask, 0, 255).astype('uint8')

    # Convert the combined mask to a ROS message and publish
    mask_msg = bridge.cv2_to_imgmsg(combined_mask, "mono8")
    output_masks_pub.publish(mask_msg)

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
    # publish_result(image_np, result)
    # publish_mask(result)

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