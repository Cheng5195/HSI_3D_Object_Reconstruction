# 1_Feature_Matching_All.py
import cv2
import numpy as np
import os
import json
from config import CommonConfig
from image_loader import BaseImageLoader
from feature_matcher import FeatureMatcher
from transformation_estimator import TransformationEstimator
from user_interface import UserInterface

class Config(CommonConfig):
    HSI_PARAMS = {'start_wavelength': 400, 'end_wavelength': 800, 'step': 5}
    @staticmethod
    def get_hsi_folder_path(object_name, frame_num):
        return f'Image Dataset/{object_name}/HSI/400_800nm/frame{frame_num}'
    @staticmethod
    def get_output_dir(object_name):
        return f'transformation/all/{object_name}'

class ImageLoader(BaseImageLoader):
    @staticmethod
    def load_hyperspectral_bands(folder_path):
        bands = []
        params = Config.HSI_PARAMS
        for wavelength in range(params['start_wavelength'], params['end_wavelength'], params['step']):
            band_path = os.path.join(folder_path, f'{wavelength}nm.png')
            band = cv2.imread(band_path, cv2.IMREAD_GRAYSCALE)
            if band is None: raise ValueError(f"Could not read band image: {band_path}")
            bands.append(band)
        return bands

    @classmethod
    def load_all_data(cls, object_name, num_frames):
        hsi_bands_list, depth_images, masks = [], [], []
        print("Loading images...")
        for frame_num in range(1, num_frames + 1):
            hsi_folder = Config.get_hsi_folder_path(object_name, frame_num)
            hsi_bands = cls.load_hyperspectral_bands(hsi_folder)
            hsi_bands_list.append(hsi_bands)

            depth_path = Config.get_depth_path(object_name, frame_num)
            depth_img = cls.load_depth_image(depth_path)
            depth_images.append(depth_img)

            json_path = Config.get_mask_json_path(object_name, frame_num)
            roi_points = cls.load_roi_from_json(json_path)
            mask = cls.create_mask(hsi_bands[0].shape, roi_points, depth_img)
            masks.append(mask)
        return hsi_bands_list, depth_images, masks

class TransformationProcessor:
    def __init__(self, object_name, distance_threshold):
        self.object_config = Config.OBJECT_CONFIGS[object_name]
        self.num_frames = self.object_config['num_frames']
        self.output_dir = Config.get_output_dir(self.object_config['name'])
        self.matcher = FeatureMatcher(distance_threshold=distance_threshold)
        self.fallback_transformation = np.identity(4)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image_pair(self, curr_frame, next_frame, hsi_bands_list, depth_images, masks):
        curr_idx, next_idx = curr_frame - 1, next_frame - 1
        print(f"\nProcessing pair {curr_frame} -> {next_frame}")

        raw_matches = self.matcher.match_features(hsi_bands_list[curr_idx], hsi_bands_list[next_idx], masks[curr_idx], masks[next_idx])
        filtered_matches = self.matcher.filter_duplicate_matches(raw_matches)

        min_matches = Config.FEATURE_PARAMS['min_matches']
        if len(filtered_matches) < min_matches: return self.fallback_transformation, 0.0, 0

        points_3d_1, points_3d_2, valid_3d_matches = TransformationEstimator.convert_to_3d_points(filtered_matches, depth_images[curr_idx], depth_images[next_idx])
        if len(valid_3d_matches) < min_matches: return self.fallback_transformation, 0.0, 0
        
        matrix, fitness = TransformationEstimator.compute_transformation_matrix(points_3d_1, points_3d_2)
        return matrix, fitness, len(valid_3d_matches)

    def run(self):
        print(f"\nProcessing {self.object_config['name']}...")
        hsi_bands_list, depth_images, masks = ImageLoader.load_all_data(self.object_config['name'], self.num_frames)
        
        print("\nProcessing image pairs...")
        for i in range(1, self.num_frames + 1):
            next_i = 1 if i == self.num_frames else i + 1
            matrix, fitness, count = self.process_image_pair(i, next_i, hsi_bands_list, depth_images, masks)

            output_path = Config.get_transformation_path(self.output_dir, i, next_i)
            np.savez(output_path, transformation=matrix, fitness=fitness, valid_matches_count=count)
            print(f"  Saved transformation: fitness={fitness:.4f}, matches={count}")
        
        print("\nAll transformations computed and saved.")

def main():
    # ... (main function is identical to the one above)
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