import cv2
import numpy as np
import json
import os
# Import CommonConfig and MatchInfo from the common configuration file
from config import CommonConfig, MatchInfo

# Define configuration specific to this script
class Config(CommonConfig):
    """Configuration parameters management class (for mean grayscale image matching)"""

    # File path methods
    @staticmethod
    def get_grayscale_path(object_name, frame_num):
        return f'Image Dataset/{object_name}/HSI/grayscale/image{frame_num}.png'

    @staticmethod
    def get_output_dir(object_name):
        return f'transformation/Mean/{object_name}'


class ImageLoader:
    """Image loading and preprocessing class"""

    @staticmethod
    def load_grayscale_image(path):
        """Load grayscale image"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read grayscale image: {path}")
        return img

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
        if min_depth is None:
            min_depth = Config.DEPTH_PARAMS['min_depth']
        if max_depth is None:
            max_depth = Config.DEPTH_PARAMS['max_depth']

        # Create ROI mask
        roi_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [points], 255)

        if depth_image is None:
            return roi_mask

        # Create depth mask
        depth_mask = np.zeros_like(roi_mask)
        scale_factor = Config.DEPTH_PARAMS['scale_factor']
        valid_depth = (depth_image > min_depth * scale_factor) & (depth_image < max_depth * scale_factor)
        depth_mask[valid_depth] = 255

        # Combine ROI and depth masks
        return cv2.bitwise_and(roi_mask, depth_mask)

    @classmethod
    def load_all_data(cls, object_name, num_frames):
        """Load image data for all frames"""
        grayscale_images = []
        depth_images = []
        masks = []

        print("Loading images...")
        for frame_num in range(1, num_frames + 1):
            # Load grayscale image
            gray_path = Config.get_grayscale_path(object_name, frame_num)
            gray_img = cls.load_grayscale_image(gray_path)
            grayscale_images.append(gray_img)

            # Load depth image
            depth_path = Config.get_depth_path(object_name, frame_num)
            depth_img = cls.load_depth_image(depth_path)
            depth_images.append(depth_img)

            # Create mask
            json_path = Config.get_mask_json_path(object_name, frame_num)
            roi_points = cls.load_roi_from_json(json_path)
            mask = cls.create_mask(gray_img.shape, roi_points, depth_img)
            masks.append(mask)

        return grayscale_images, depth_images, masks


class FeatureMatcher:
    """Feature extraction and matching class"""

    def __init__(self, distance_threshold=None):
        """Initialize feature matcher"""
        params = Config.FEATURE_PARAMS
        self.sift = cv2.SIFT_create(
            nfeatures=params['nfeatures'],
            contrastThreshold=params['contrastThreshold'],
            edgeThreshold=params['edgeThreshold'],
        )
        self.matcher = cv2.BFMatcher()
        self.distance_threshold = distance_threshold if distance_threshold is not None else params['default_distance_threshold']
        self.ratio_threshold = params['ratio_threshold']
        self.min_matches = params['min_matches']

    def match_features(self, img1, img2, mask1=None, mask2=None):
        """Perform feature extraction and matching"""
        # Detect keypoints and compute descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, mask1)
        kp2, des2 = self.sift.detectAndCompute(img2, mask2)

        # Check if descriptors are sufficient
        if des1 is None or des2 is None or len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return []

        # KNN matching
        matches = self.matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(MatchInfo(kp1[m.queryIdx], kp2[m.trainIdx], m.distance))

        return good_matches

    def filter_duplicate_matches(self, matches):
        """Filter duplicate matching points"""
        if not matches:
            return []

        # Sort matching points by distance
        sorted_matches = sorted(matches, key=lambda x: x.distance)

        # For tracking used points
        used_points1, used_points2 = set(), set()
        filtered_matches = []

        # Filter duplicate points
        for match in sorted_matches:
            pt1 = match.kp1.pt
            pt2 = match.kp2.pt

            # Check if duplicate point
            is_duplicate = False
            for used_pt1 in used_points1:
                if np.linalg.norm(np.array(pt1) - np.array(used_pt1)) < self.distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                for used_pt2 in used_points2:
                    if np.linalg.norm(np.array(pt2) - np.array(used_pt2)) < self.distance_threshold:
                        is_duplicate = True
                        break

            # If not duplicate, keep it
            if not is_duplicate:
                filtered_matches.append(match)
                used_points1.add(pt1)
                used_points2.add(pt2)

        return filtered_matches


class TransformationEstimator:
    """3D transformation estimation class"""

    @staticmethod
    def convert_to_3d_points(matches, depth_image1, depth_image2):
        """Convert 2D matching points to 3D point pairs"""
        # Get camera parameters
        fx = Config.CAMERA_PARAMS['fx']
        fy = Config.CAMERA_PARAMS['fy']
        cx = Config.CAMERA_PARAMS['cx']
        cy = Config.CAMERA_PARAMS['cy']
        scale_factor = Config.DEPTH_PARAMS['scale_factor']

        points_3d_1, points_3d_2 = [], []
        valid_matches = []

        for match in matches:
            # Get 2D point coordinates
            u1, v1 = map(int, match.kp1.pt)
            u2, v2 = map(int, match.kp2.pt)

            # Check if points are within image boundaries
            if (0 <= u1 < depth_image1.shape[1] and 0 <= v1 < depth_image1.shape[0] and
                0 <= u2 < depth_image2.shape[1] and 0 <= v2 < depth_image2.shape[0]):

                # Get depth values and convert to meters
                z1 = depth_image1[v1, u1] / scale_factor
                z2 = depth_image2[v2, u2] / scale_factor

                # Check if depth values are valid
                if z1 > 0 and z2 > 0:
                    # Calculate 3D point coordinates
                    x1 = (u1 - cx) * z1 / fx
                    y1 = (v1 - cy) * z1 / fy
                    x2 = (u2 - cx) * z2 / fx
                    y2 = (v2 - cy) * z2 / fy

                    points_3d_1.append([x1, y1, z1])
                    points_3d_2.append([x2, y2, z2])
                    valid_matches.append(match)

        return np.array(points_3d_1), np.array(points_3d_2), valid_matches

    @staticmethod
    def compute_rigid_transform_svd(src_pts, dst_pts):
        """Compute rigid transformation using SVD"""
        # Calculate centroids
        centroid_src = np.mean(src_pts, axis=0)
        centroid_dst = np.mean(dst_pts, axis=0)

        # Center point clouds
        src_centered = src_pts - centroid_src
        dst_centered = dst_pts - centroid_dst

        # Compute covariance matrix
        H = src_centered.T @ dst_centered

        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)

        # Calculate rotation matrix
        R = Vt.T @ U.T

        # Handle mirror reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Calculate translation vector
        t = centroid_dst - R @ centroid_src

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @classmethod
    def ransac_3d_rigid_transform(cls, pts1, pts2, threshold=None, max_iters=None):
        """Estimate 3D rigid transformation using RANSAC"""
        # Get RANSAC parameters
        if threshold is None:
            threshold = Config.RANSAC_PARAMS['threshold']
        if max_iters is None:
            max_iters = Config.RANSAC_PARAMS['max_iters']

        min_inliers = Config.RANSAC_PARAMS['min_inliers']
        success_ratio = Config.RANSAC_PARAMS['success_ratio']

        # Check if there are enough points
        if len(pts1) < min_inliers:
            return None, np.zeros(len(pts1), dtype=bool)

        # RANSAC iterations
        best_inlier_count = 0
        best_inliers_mask = np.zeros(len(pts1), dtype=bool)
        best_trans = np.eye(4)

        for _ in range(max_iters):
            # Randomly select points
            sample_idx = np.random.choice(len(pts1), min_inliers, replace=False)

            # Estimate transformation using sample points
            T_est = cls.compute_rigid_transform_svd(pts1[sample_idx], pts2[sample_idx])

            # Apply transformation to all points
            pts1_homo = np.hstack([pts1, np.ones((len(pts1), 1))])
            transformed = (T_est @ pts1_homo.T).T

            # Calculate errors
            errors = np.linalg.norm(transformed[:, :3] - pts2, axis=1)

            # Find inliers
            inliers_mask = errors < threshold
            inlier_count = np.sum(inliers_mask)

            # Update best result
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers_mask = inliers_mask
                best_trans = T_est

                # If inlier ratio is high enough, terminate early
                if inlier_count > success_ratio * len(pts1):
                    break

        # Refine transformation using all inliers
        if best_inlier_count >= min_inliers:
            refined_src = pts1[best_inliers_mask]
            refined_dst = pts2[best_inliers_mask]
            best_trans = cls.compute_rigid_transform_svd(refined_src, refined_dst)

        return best_trans, best_inliers_mask

    @classmethod
    def compute_transformation_matrix(cls, points_3d_1, points_3d_2, threshold=None):
        """Compute transformation matrix between 3D point pairs"""
        min_inliers = Config.RANSAC_PARAMS['min_inliers']

        # Check if there are enough points
        if len(points_3d_1) < min_inliers:
            return np.eye(4), 0.0

        # Estimate transformation using RANSAC
        T_est, inliers_mask = cls.ransac_3d_rigid_transform(points_3d_1, points_3d_2, threshold)

        # Check inlier count
        if np.sum(inliers_mask) < min_inliers:
            return np.eye(4), 0.0

        # Calculate fitness score
        pts1_homo = np.hstack([points_3d_1, np.ones((len(points_3d_1), 1))])
        transformed = (T_est @ pts1_homo.T).T

        if threshold is None:
            threshold = Config.RANSAC_PARAMS['threshold']

        errors = np.linalg.norm(transformed[:, :3] - points_3d_2, axis=1)
        final_inliers = errors < threshold
        fitness = np.sum(final_inliers) / len(points_3d_1)

        return T_est, fitness


class UserInterface:
    """User interaction interface class"""

    @staticmethod
    def select_object():
        """Select object to process"""
        print("\nAvailable objects:")
        for i, (key, config) in enumerate(Config.OBJECT_CONFIGS.items(), 1):
            print(f"{i}. {config['name']} ({config['num_frames']} frames)")

        while True:
            try:
                choice = int(input("\nSelect object number (1-4): "))
                if 1 <= choice <= 4:
                    return list(Config.OBJECT_CONFIGS.keys())[choice - 1]
                print("Invalid choice. Please select a number between 1-4.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    @staticmethod
    def select_distance_threshold():
        """Function to adjust distance threshold"""
        default_threshold = Config.FEATURE_PARAMS['default_distance_threshold']
        print(f"\nCurrent distance threshold for filtering duplicate matches is: {default_threshold}")

        adjust = input("Do you want to adjust the distance threshold? (y/n): ").lower().strip()

        if adjust == 'y' or adjust == 'yes':
            while True:
                try:
                    threshold = float(input("Enter new distance threshold (0-10): "))
                    if 0 <= threshold <= 10:
                        print(f"Distance threshold set to: {threshold}")
                        return threshold
                    print("Invalid threshold. Please enter a value between 0 and 10.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print(f"Using default distance threshold: {default_threshold}")
            return default_threshold


class TransformationProcessor:
    """Transformation processing class"""

    def __init__(self, object_name, distance_threshold):
        """Initialize processor"""
        self.object_name = object_name
        self.distance_threshold = distance_threshold
        self.object_config = Config.OBJECT_CONFIGS[object_name]
        self.num_frames = self.object_config['num_frames']
        self.output_dir = Config.get_output_dir(self.object_config['name'])
        self.matcher = FeatureMatcher(distance_threshold=distance_threshold)
        self.fallback_transformation = np.identity(4)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image_pair(self, curr_frame, next_frame, grayscale_images, depth_images, masks):
        """Process image pair and compute transformation matrix"""
        # Frame index starts from 0, while frame number starts from 1
        curr_idx = curr_frame - 1
        next_idx = next_frame - 1

        print(f"\nProcessing pair {curr_frame} -> {next_frame}")

        # Feature matching and filtering
        raw_matches = self.matcher.match_features(
            grayscale_images[curr_idx],
            grayscale_images[next_idx],
            masks[curr_idx],
            masks[next_idx]
        )
        filtered_matches = self.matcher.filter_duplicate_matches(raw_matches)

        min_matches = Config.FEATURE_PARAMS['min_matches']
        if len(filtered_matches) < min_matches:
            return self.fallback_transformation, 0.0, 0

        # 3D point conversion and transformation matrix calculation
        points_3d_1, points_3d_2, valid_3d_matches = TransformationEstimator.convert_to_3d_points(
            filtered_matches,
            depth_images[curr_idx],
            depth_images[next_idx]
        )

        if len(valid_3d_matches) < min_matches:
            return self.fallback_transformation, 0.0, 0

        transformation_matrix, fitness = TransformationEstimator.compute_transformation_matrix(
            points_3d_1, points_3d_2
        )

        return transformation_matrix, fitness, len(valid_3d_matches)

    def run(self):
        """Run processing workflow"""
        print(f"\nProcessing {self.object_config['name']} with {self.num_frames} frames...")
        print(f"Using distance threshold: {self.distance_threshold}")

        # Load all image data
        grayscale_images, depth_images, masks = ImageLoader.load_all_data(
            self.object_config['name'], self.num_frames
        )

        # Process all image pairs
        print("\nProcessing image pairs...")
        edge_info_list = []

        for i in range(1, self.num_frames + 1):
            # Determine next frame (loop back to first frame)
            next_i = 1 if i == self.num_frames else i + 1

            # Process image pair
            transformation_matrix, fitness, valid_matches_count = self.process_image_pair(
                i, next_i, grayscale_images, depth_images, masks
            )

            # Save results
            edge_info = {
                'source': i,
                'target': next_i,
                'transformation': transformation_matrix,
                'fitness': fitness,
                'valid_matches_count': valid_matches_count,
                'distance_threshold': self.distance_threshold
            }
            edge_info_list.append(edge_info)

            output_path = Config.get_transformation_path(self.output_dir, i, next_i)
            np.savez(output_path,
                     transformation=transformation_matrix,
                     fitness=fitness,
                     valid_matches_count=valid_matches_count,
                     distance_threshold=self.distance_threshold)

            print(f"  Saved transformation matrix: fitness={fitness:.4f}, matches={valid_matches_count}")

        print("\nAll transformations have been computed and saved.")
        print(f"Distance threshold used: {self.distance_threshold}")


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
