#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import mmcv
import torch
import heapq
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Float32, Int8
from cv_bridge import CvBridge
import time
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import load_config, get_input_shape
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

# Initialize CvBridge for converting ROS images to OpenCV format
bridge = CvBridge()

# Configure the log file path with datetime suffix
log_file = f"/home/kevin/Documents/test_ros/log/instance_segment_log/seg_cow_fodder_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
handler = TimedRotatingFileHandler(
    log_file, when="D", interval=1, backupCount=30
)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Set up the logger
logger = logging.getLogger('rospy_logger')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Load your model and configuration once
deploy_cfg = '/home/kevin/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py'
device = 'cuda'

# model update at 20241017
# model_cfg = '/home/kevin/mmdetection/configs/cow/rtmdet-ins_tiny_1xb4-500e_cow_update_20241017.py'
# backend_model = ['/home/kevin/mmdeploy/mmdeploy_models/cow/rtmdet-ins_tiny/end2end.engine']

# model update at 20241202
model_cfg = '/home/kevin/mmdetection/configs/cow/rtmdet-ins_tiny_1xb4-500e_cow_update_20241202.py'
backend_model = ['/home/kevin/mmdeploy/mmdeploy_models/cow/rtmdet-ins_tiny_1203/end2end.engine']

output_image_pub = rospy.Publisher('/segmentation_result/image', Image, queue_size=60)
output_bboxes_pub = rospy.Publisher('/segmentation_result/bboxes', Float32MultiArray, queue_size=60)
output_labels_pub = rospy.Publisher('/segmentation_result/labels', Int32MultiArray, queue_size=60)
output_scores_pub = rospy.Publisher('/segmentation_result/scores', Float32MultiArray, queue_size=60)
output_masks_pub = rospy.Publisher('/segmentation_result/masks', Image, queue_size=60)  # Publish masks as images
fodder_bunk_ratio_pub = rospy.Publisher('fodder_bunk_ratio', Float32, queue_size=10)
clostest_depth_cow_pub = rospy.Publisher('clostest_depth_cow', Float32, queue_size=10)
panel_angle_pub = rospy.Publisher('panel_angle', Int8, queue_size=10)

# Store centroids and depth image globally for use across callbacks
centroids = []
# Initialize the min-heap to store depths
depth_heap = []
# Initialize necessary variables
depths = None
combined_fodder_mask = None
combined_bunk_mask = None
smallest_depth = None
fodder_bunk_ratio = None

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

def extract_mask(result):
    # Extract cow information
    # Assuming your instance segmentation results are stored as follows:
    pred_instances = result.pred_instances
    masks = pred_instances.masks  # Tensor of shape (N, H, W) where N is the number of instances
    labels = pred_instances.labels  # Tensor of shape (N,)
    bboxes = pred_instances.bboxes  # Tensor of shape (N, 4)
    scores = pred_instances.scores  # Tensor of shape (N,)
    return masks, labels ,bboxes, scores

def extract_special_mask(image_np, result, is_visualize = False, cls_label = 2, score_threshold = 0.5):# default extract cow mask (cls_label == 2) and not visialuze
    masks, labels ,bboxes, scores = extract_mask(result)
    # Get the mask indices where label == 2 (i.e., cow instances)
    indices = ((labels == cls_label) & (scores >= score_threshold)).nonzero(as_tuple=True)[0]  # This gives you the indices of "cow" instances
    # Filter the masks for cows
    indices_masks = masks[indices]

    # Optionally, filter other attributes as well if needed
    indices_bboxes = bboxes[indices]
    indices_scores = scores[indices]
    # Visualize the masks and bounding boxes if is_visualize is set to True
    if is_visualize:
        # Clone the original image to avoid modifying it directly
        visual_image = image_np.copy()
        
        for i, mask in enumerate(indices_masks):
            # Convert the mask tensor to a binary mask for OpenCV
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255

            # Apply a random color for each mask
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            
            # Overlay the mask on the original image
            visual_image[mask_np > 0] = 0.5 * visual_image[mask_np > 0] + 0.5 * np.array(color)

            # Draw bounding box for each detected cow
            bbox = indices_bboxes[i].cpu().numpy().astype(int)
            cv2.rectangle(visual_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Optionally, add score text near the bounding box
            score_text = f"{indices_scores[i].item():.2f}"
            cv2.putText(visual_image, score_text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show the image
        cv2.imshow("Cow Masks and Bounding Boxes", visual_image)
        cv2.waitKey(0)
    
    return indices_masks, indices_bboxes, indices_scores

def calculate_cow_centroids(image_np, result, is_visualize = False):
    # Extract cow information
    cow_masks, _, _ = extract_special_mask(image_np, result)
 
    # Assuming cow_masks is a tensor of shape (N, H, W), where N is the number of cow instances
    global centroids
    centroids.clear()  # Clear previous centroids

    for mask in cow_masks:
        # Get the non-zero indices (y, x coordinates) of the mask
        y_indices, x_indices = torch.nonzero(mask, as_tuple=True)
        
        # Calculate the mean of x and y indices to find the centroid
        if y_indices.numel() > 0 and x_indices.numel() > 0:  # Ensure there are non-zero values
            centroid_x = x_indices.float().mean().item()
            centroid_y = y_indices.float().mean().item()
            centroids.append((centroid_x, centroid_y))
    # Visualize the centroids if is_visualize is set to True
    if is_visualize:
        # Clone the original image for visualization
        visual_image = image_np.copy()

        # Draw each centroid as a small circle on the visual image
        for centroid_x, centroid_y in centroids:
            # Convert centroid coordinates to integer for OpenCV
            center = (int(centroid_x), int(centroid_y))
            color = (0, 255, 0)  # Green color for the centroids
            cv2.circle(visual_image, center, radius=5, color=color, thickness=-1)  # Filled circle

        # Display the image with centroids
        cv2.imshow("Centroids on Masks", visual_image)
        cv2.waitKey(0)

    return centroids

def extract_fodder_bunk_mask(image_np, result):
    global combined_fodder_mask, combined_bunk_mask
    
    # Extract masks
    bunk_mask, _, _, = extract_special_mask(image_np, result, False, 1, 0.5)
    fodder_mask, _, _ = extract_special_mask(image_np, result, False, 3, 0.3)

    # Convert to NumPy arrays
    fodder_mask = fodder_mask.cpu().numpy() if fodder_mask is not None else None
    bunk_mask = bunk_mask.cpu().numpy() if bunk_mask is not None else None

    # Handle fodder mask
    if fodder_mask is not None and fodder_mask.size > 0:
        combined_fodder_mask = np.zeros_like(fodder_mask[0], dtype=np.uint8)
        for mask in fodder_mask:
            combined_fodder_mask = np.logical_or(combined_fodder_mask, mask).astype(np.uint8)
    else:
        combined_fodder_mask = None

    # Handle bunk mask
    if bunk_mask is not None and bunk_mask.size > 0:
        combined_bunk_mask = np.zeros_like(bunk_mask[0], dtype=np.uint8)
        for mask in bunk_mask:
            combined_bunk_mask = np.logical_or(combined_bunk_mask, mask).astype(np.uint8)
    else:
        combined_bunk_mask = None

    return combined_fodder_mask, combined_bunk_mask

# def extract_fodder_bunk_mask_tensor(image_np, result):
#     global fodder_mask, bunk_mask
#     bunk_mask, _, _, = extract_special_mask(image_np, result, False, 1, 0.5)
#     fodder_mask, _, _ = extract_special_mask(image_np, result, False, 3, 0.3)
#     return fodder_mask, bunk_mask

def get_depth_weight_tensor(depth_value):
    # 根据深度值的范围，设置权重
    return torch.where(depth_value < 4000, torch.tensor(1.0), torch.where(depth_value < 6000, torch.tensor(0.8), torch.tensor(0.0)))

def get_depth_weight(depth_value):
    # 根据深度值的范围，设置权重
    return np.where(depth_value < 4000, 1.0, np.where(depth_value < 6000, 0.8, 0))

def calculate_fodder_bunk_ratio(fodder_mask, bunk_mask, depth_image):
    # ensure bool
    fodder_mask_bool = fodder_mask.astype(bool)
    bunk_mask_bool = bunk_mask.astype(bool)

    # extract depth value
    fodder_depth = depth_image[fodder_mask_bool]
    bunk_depth = depth_image[bunk_mask_bool]
    # Remove nan or infinite values if they exist
    fodder_depth = fodder_depth[np.isfinite(fodder_depth)]
    bunk_depth = bunk_depth[np.isfinite(bunk_depth)]

    # 累加深度值
    fodder_depth_sum = np.sum(fodder_depth * fodder_depth * get_depth_weight(fodder_depth))
    bunk_depth_sum = np.sum(bunk_depth * bunk_depth * get_depth_weight(bunk_depth))

    if fodder_depth is None or bunk_depth_sum is None or  bunk_depth_sum == 0:
        return 0

    # Calculate the ratio
    fodder_bunk_ratio = fodder_depth_sum / bunk_depth_sum
    fodder_bunk_ratio_pub.publish(fodder_bunk_ratio)
    # Print or return the ratio
    return fodder_bunk_ratio

def calculate_clostest_depth_cow(depths, depth_msg):  
    if centroids:
        depth_heap.clear()
        for u, v in centroids:
            # Convert centroid coordinates to integers for indexing
            u = int(u)
            v = int(v)
            Idx = u + depth_msg.width * v
            depth = depths[Idx]
            # Push the depth into the min-heap
            heapq.heappush(depth_heap, depth)
    
    # Extract the smallest depth from the min-heap if it's not empty
    if depth_heap:
        smallest_depth = heapq.heappop(depth_heap)
        return smallest_depth

    return None

def judge_panel_angle(fodder_bunk_ratio, smallest_depth):
    if smallest_depth > 3800 and fodder_bunk_ratio < 1:
        panel_angle_pub.publish(90)
        return 90
    else:
        panel_angle_pub.publish(30)
        return 30

def depthCallback(depth_msg):
    global depths, smallest_depth, fodder_bunk_ratio, combined_fodder_mask, combined_bunk_mask
    # Convert the raw data to a NumPy array of floats
    depths = np.frombuffer(depth_msg.data, dtype=np.float32)
    depth_image = depths.reshape(depth_msg.height, depth_msg.width)

    # Make a writable copy of depth_image for tensor conversion
    # depth_tensor = torch.from_numpy(depth_image.copy()).float()
    # depth_tensor = depth_tensor.to('cuda')   

    if centroids:
        depth_heap.clear()
        for u, v in centroids:
            # Convert centroid coordinates to integers for indexing
            u = int(u)
            v = int(v)
            Idx = u + depth_msg.width * v
            depth = depths[Idx]
            # Push the depth into the min-heap
            heapq.heappush(depth_heap, depth)
    
    # Extract the smallest depth from the min-heap if it's not empty
    if depth_heap:
        smallest_depth = heapq.heappop(depth_heap)
        # Ensure smallest_depth is finite before publishing
        if np.isfinite(smallest_depth):
            clostest_depth_cow_pub.publish(smallest_depth)
            rospy.loginfo(f"The Closest depth of the cow: {smallest_depth / 1000} m")
            logger.info(f"The Closest depth of the cow: {smallest_depth / 1000} m")
        else:
            rospy.logwarn("Encountered non-finite closest depth value.")
            logger.warning("Encountered non-finite closest depth value.")

    if combined_fodder_mask is not None and combined_bunk_mask is not None:
        fodder_bunk_ratio = calculate_fodder_bunk_ratio(combined_fodder_mask, combined_bunk_mask, depth_image)
        rospy.loginfo(f"Fodder to Bunk Ratio {fodder_bunk_ratio}")
        logger.info(f"Fodder to Bunk Ratio {fodder_bunk_ratio}")

    if smallest_depth is not None and fodder_bunk_ratio is not None:
        panel_angle = judge_panel_angle(fodder_bunk_ratio, smallest_depth)
        rospy.loginfo(f"Panel Angle should be {panel_angle}")
        logger.info(f"Panel Angle should be {panel_angle}")
    rospy.loginfo('-'*60)

# Callback function to process each received image
def image_callback(ros_image):
    # Convert ROS image to OpenCV format (BGR)
    image_np = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    # Run inference
    start_time = time.time()
    result = inference_image(image_np)
    process_time = time.time() - start_time
    rospy.loginfo(f"Inference time: {process_time:.4f}s")
    logger.info(f"Inference time: {process_time:.4f}s")

    # Extract Mask
    fodder_mask, bunk_mask = extract_fodder_bunk_mask(image_np, result)
    calculate_cow_centroids(image_np, result)

def main():
    # Initialize ROS node
    rospy.init_node('zed_tensorRT_segmenter', anonymous=True)

    # Load the TensorRT model
    init_model()

    rospy.loginfo('-'*60)
    # Subscribe to the ZED camera image and depth topic
    rospy.Subscriber("/zed2i/zed_node/rgb_raw/image_raw_color", Image, image_callback)
    rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, depthCallback)
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()