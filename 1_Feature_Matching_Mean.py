# 1_Feature_Matching_Mean.py
import cv2
import numpy as np
import os
from config import CommonConfig
from image_loader import BaseImageLoader
from feature_matcher import FeatureMatcher
from transformation_estimator import TransformationEstimator
from user_interface import UserInterface

class Config(CommonConfig):
    @staticmethod
    def get_grayscale_path(object_name, frame_num):
        return f'Image Dataset/{object_name}/HSI/grayscale/image{frame_num}.png'

    @staticmethod
    def get_output_dir(object_name):
        return f'transformation/Mean/{object_name}'

class ImageLoader(BaseImageLoader):
    @staticmethod
    def load_grayscale_image(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f"Could not read grayscale image: {path}")
        return img

    @classmethod
    def load_all_data(cls, object_name, num_frames):
        images, depth_images, masks = [], [], []
        print("Loading images...")
        for frame_num in range(1, num_frames + 1):
            gray_path = Config.get_grayscale_path(object_name, frame_num)
            gray_img = cls.load_grayscale_image(gray_path)
            images.append(gray_img)

            depth_path = Config.get_depth_path(object_name, frame_num)
            depth_img = cls.load_depth_image(depth_path)
            depth_images.append(depth_img)

            json_path = Config.get_mask_json_path(object_name, frame_num)
            roi_points = cls.load_roi_from_json(json_path)
            mask = cls.create_mask(gray_img.shape, roi_points, depth_img)
            masks.append(mask)
        return images, depth_images, masks

class TransformationProcessor:
    def __init__(self, object_name, distance_threshold):
        self.object_config = Config.OBJECT_CONFIGS[object_name]
        self.num_frames = self.object_config['num_frames']
        self.output_dir = Config.get_output_dir(self.object_config['name'])
        self.matcher = FeatureMatcher(distance_threshold=distance_threshold)
        self.fallback_transformation = np.identity(4)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image_pair(self, curr_frame, next_frame, images, depth_images, masks):
        curr_idx, next_idx = curr_frame - 1, next_frame - 1
        print(f"\nProcessing pair {curr_frame} -> {next_frame}")

        # Note: We wrap the single image in a list to match the FeatureMatcher's expectation
        raw_matches = self.matcher.match_features([images[curr_idx]], [images[next_idx]], masks[curr_idx], masks[next_idx])
        filtered_matches = self.matcher.filter_duplicate_matches(raw_matches)

        min_matches = Config.FEATURE_PARAMS['min_matches']
        if len(filtered_matches) < min_matches: return self.fallback_transformation, 0.0, 0

        points_3d_1, points_3d_2, valid_3d_matches = TransformationEstimator.convert_to_3d_points(filtered_matches, depth_images[curr_idx], depth_images[next_idx])
        if len(valid_3d_matches) < min_matches: return self.fallback_transformation, 0.0, 0

        matrix, fitness = TransformationEstimator.compute_transformation_matrix(points_3d_1, points_3d_2)
        return matrix, fitness, len(valid_3d_matches)

    def run(self):
        print(f"\nProcessing {self.object_config['name']}...")
        images, depth_images, masks = ImageLoader.load_all_data(self.object_config['name'], self.num_frames)

        print("\nProcessing image pairs...")
        for i in range(1, self.num_frames + 1):
            next_i = 1 if i == self.num_frames else i + 1
            matrix, fitness, count = self.process_image_pair(i, next_i, images, depth_images, masks)
            
            output_path = Config.get_transformation_path(self.output_dir, i, next_i)
            np.savez(output_path, transformation=matrix, fitness=fitness, valid_matches_count=count)
            print(f"  Saved transformation: fitness={fitness:.4f}, matches={count}")

        print("\nAll transformations computed and saved.")

def main():
    try:
        object_key = UserInterface.select_object()
        distance_threshold = UserInterface.select_distance_threshold()
        processor = TransformationProcessor(object_key, distance_threshold)
        processor.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()