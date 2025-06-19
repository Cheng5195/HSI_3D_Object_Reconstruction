import os
from collections import namedtuple
import numpy as np
import cv2

# Define MatchInfo namedtuple, as it is used across all files
MatchInfo = namedtuple('MatchInfo', ['kp1', 'kp2', 'distance'])

class CommonConfig:
    """Common configuration parameters management class"""

    # Camera parameters
    CAMERA_PARAMS = {
        'fx': 747.428696704179,
        'fy': 707.986035101762,
        'cx': 582.360488889745,
        'cy': 414.993062461901
    }

    # Feature extraction and matching parameters
    FEATURE_PARAMS = {
        'nfeatures': 2000,
        'contrastThreshold': 0.01,
        'edgeThreshold': 10,
        'ratio_threshold': 0.75,
        'default_distance_threshold': 0,
        'min_matches': 3
    }

    # RANSAC parameters
    RANSAC_PARAMS = {
        'threshold': 0.015,
        'max_iters': 1000,
        'success_ratio': 0.9,
        'min_inliers': 3
    }

    # Depth parameters
    DEPTH_PARAMS = {
        'min_depth': 0.3,
        'max_depth': 0.6,
        'scale_factor': 1000.0  # Millimeter to meter scale factor
    }

    # Object configurations
    OBJECT_CONFIGS = {
        'chips_can': {'name': 'Chips can', 'num_frames': 26},
        'cracker_box': {'name': 'Cracker box', 'num_frames': 25},
        'power_drill': {'name': 'Power drill', 'num_frames': 29},
        'wood_block': {'name': 'Wood block', 'num_frames': 24}
    }

    # File path methods - these can now be more generic as they are inherited by method-specific Config classes
    @staticmethod
    def get_depth_path(object_name, frame_num):
        return f'Image Dataset/{object_name}/HSI/depth/transformed_undistorted_depth_{frame_num}.png'

    @staticmethod
    def get_mask_json_path(object_name, frame_num):
        return f'Image Dataset/{object_name}/HSI/mask/undistorted_hsi_{frame_num}.json'

    @staticmethod
    def get_transformation_path(output_dir, source_frame, target_frame):
        return f'{output_dir}/transformation_{source_frame}_{target_frame}.npz'