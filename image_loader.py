# image_loader.py
import cv2
import numpy as np
import json
from config import CommonConfig

class BaseImageLoader:
    """Base class for image loading and preprocessing."""

    @staticmethod
    def load_depth_image(path):
        """Load depth image"""
        depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise ValueError(f"Could not read depth image: {path}")
        return depth_img

    @staticmethod
    def load_roi_from_json(json_path):
        """Load ROI points from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return np.array(data['shapes'][0]['points'], dtype=np.int32)

    @staticmethod
    def create_mask(image_shape, points, depth_image=None, min_depth=None, max_depth=None):
        """Create mask for ROI and depth information"""
        if min_depth is None: min_depth = CommonConfig.DEPTH_PARAMS['min_depth']
        if max_depth is None: max_depth = CommonConfig.DEPTH_PARAMS['max_depth']

        roi_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [points], 255)

        if depth_image is None:
            return roi_mask

        depth_mask = np.zeros_like(roi_mask)
        scale_factor = CommonConfig.DEPTH_PARAMS['scale_factor']
        valid_depth = (depth_image > min_depth * scale_factor) & (depth_image < max_depth * scale_factor)
        depth_mask[valid_depth] = 255

        return cv2.bitwise_and(roi_mask, depth_mask)