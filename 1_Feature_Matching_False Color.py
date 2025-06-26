# 1_Feature_Matching_False_Color.py
import cv2
import numpy as np
import os
import json
from config import CommonConfig
from image_loader import BaseImageLoader
from feature_matcher import FeatureMatcher
from transformation_estimator import TransformationEstimator
from user_interface import UserInterface

# Define configuration specific to this script
class Config(CommonConfig):
    """Configuration parameters management class (for false color matching)"""

    # Selected wavelengths for false color image
    WAVELENGTHS = [435, 545, 700]

    # File path methods
    @staticmethod
    def get_hsi_folder_path(object_name, frame_num):
        return f'Image Dataset/{object_name}/HSI/400_800nm/frame{frame_num}'

    @staticmethod
    def get_output_dir(object_name):
        return f'transformation/False Color/{object_name}'

class ImageLoader(BaseImageLoader):
    """Image loading and preprocessing class"""

    @staticmethod
    def load_hyperspectral_bands(folder_path, wavelengths=None):
        """Load hyperspectral image bands of specified wavelengths"""
        # Use the wavelengths defined in the Config class if not provided
        if wavelengths is None:
            wavelengths = Config.WAVELENGTHS

        bands = []
        for wavelength in wavelengths:
            band_path = os.path.join(folder_path, f'{wavelength}nm.png')
            band = cv2.imread(band_path, cv2.IMREAD_GRAYSCALE)
            if band is None:
                raise ValueError(f"Could not read band image: {band_path}")
            bands.append(band)

        return bands

    @classmethod
    def load_all_data(cls, object_name, num_frames):
        """Load image data for all frames"""
        hsi_bands_list = []
        depth_images = []
        masks = []

        print("Loading images...")
        for frame_num in range(1, num_frames + 1):
            # Load specific hyperspectral image bands for false color
            hsi_folder = Config.get_hsi_folder_path(object_name, frame_num)
            hsi_bands = cls.load_hyperspectral_bands(hsi_folder) # This will use the specific WAVELENGTHS
            hsi_bands_list.append(hsi_bands)

            # Load depth image
            depth_path = Config.get_depth_path(object_name, frame_num)
            depth_img = cls.load_depth_image(depth_path)
            depth_images.append(depth_img)

            # Create mask from ROI and depth information
            json_path = Config.get_mask_json_path(object_name, frame_num)
            roi_points = cls.load_roi_from_json(json_path)
            mask = cls.create_mask(hsi_bands[0].shape, roi_points, depth_img)
            masks.append(mask)

        return hsi_bands_list, depth_images, masks


class TransformationProcessor:
    """Transformation processing class"""

    def __init__(self, object_name, distance_threshold):
        """Initialize processor"""
        self.object_config = Config.OBJECT_CONFIGS[object_name]
        self.num_frames = self.object_config['num_frames']
        self.output_dir = Config.get_output_dir(self.object_config['name'])
        # Initialize the unified FeatureMatcher
        self.matcher = FeatureMatcher(distance_threshold=distance_threshold)
        self.fallback_transformation = np.identity(4)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image_pair(self, curr_frame, next_frame, hsi_bands_list, depth_images, masks):
        """Process image pair and compute transformation matrix"""
        curr_idx = curr_frame - 1
        next_idx = next_frame - 1

        print(f"\nProcessing pair {curr_frame} -> {next_frame}")

        # Feature matching on the selected bands and filtering
        raw_matches = self.matcher.match_features(
            hsi_bands_list[curr_idx],
            hsi_bands_list[next_idx],
            masks[curr_idx],
            masks[next_idx]
        )
        filtered_matches = self.matcher.filter_duplicate_matches(raw_matches)

        min_matches = Config.FEATURE_PARAMS['min_matches']
        if len(filtered_matches) < min_matches:
            return np.identity(4), 0.0, 0

        # 3D point conversion and transformation matrix calculation using the common estimator
        points_3d_1, points_3d_2, valid_3d_matches = TransformationEstimator.convert_to_3d_points(
            filtered_matches,
            depth_images[curr_idx],
            depth_images[next_idx]
        )

        if len(valid_3d_matches) < min_matches:
            return np.identity(4), 0.0, 0

        transformation_matrix, fitness = TransformationEstimator.compute_transformation_matrix(
            points_3d_1, points_3d_2
        )

        return transformation_matrix, fitness, len(valid_3d_matches)

    def run(self):
        """Run processing workflow"""
        print(f"\nProcessing {self.object_config['name']} with {self.num_frames} frames...")

        # Load all image data using the specific loader
        hsi_bands_list, depth_images, masks = ImageLoader.load_all_data(
            self.object_config['name'], self.num_frames
        )

        print("\nProcessing image pairs...")
        for i in range(1, self.num_frames + 1):
            next_i = 1 if i == self.num_frames else i + 1

            # Process image pair
            transformation_matrix, fitness, valid_matches_count = self.process_image_pair(
                i, next_i, hsi_bands_list, depth_images, masks
            )

            # Save results
            output_path = Config.get_transformation_path(self.output_dir, i, next_i)
            np.savez(output_path,
                     transformation=transformation_matrix,
                     fitness=fitness,
                     valid_matches_count=valid_matches_count)

            print(f"  Saved transformation matrix: fitness={fitness:.4f}, matches={valid_matches_count}")

        print("\nAll transformations have been computed and saved.")


def main():
    try:
        # Use the common UserInterface to get user input
        object_key = UserInterface.select_object()
        distance_threshold = UserInterface.select_distance_threshold()
        processor = TransformationProcessor(object_key, distance_threshold)
        processor.run()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()