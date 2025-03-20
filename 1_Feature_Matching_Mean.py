import cv2
import numpy as np
import json
import os
from collections import namedtuple

MatchInfo = namedtuple('MatchInfo', ['kp1', 'kp2', 'distance'])

# Camera parameters
fx, fy = 747.428696704179, 707.986035101762
cx, cy = 582.360488889745, 414.993062461901

# Object configurations
OBJECT_CONFIGS = {
    'chips_can': {'name': 'Chips can', 'num_frames': 26},
    'cracker_box': {'name': 'Cracker box', 'num_frames': 25},
    'power_drill': {'name': 'Power drill', 'num_frames': 29},
    'wood_block': {'name': 'Wood block', 'num_frames': 24}
}

class FeatureMatcher:
    """Feature Matcher Class"""
    def __init__(self, distance_threshold=0):
        self.sift = cv2.SIFT_create(
            nfeatures=2000,
            contrastThreshold=0.01,
            edgeThreshold=10,
        )
        self.matcher = cv2.BFMatcher()
        self.distance_threshold = distance_threshold

    def match_features(self, img1, img2, mask1, mask2):
        """Feature extraction and matching"""
        kp1, des1 = self.sift.detectAndCompute(img1, mask1)
        kp2, des2 = self.sift.detectAndCompute(img2, mask2)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return []

        matches = self.matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        return [MatchInfo(kp1[m.queryIdx], kp2[m.trainIdx], m.distance) 
                for m in good_matches]

    def filter_duplicate_matches(self, matches):
        """Filter duplicate matching points"""
        if not matches:
            return []

        sorted_matches = sorted(matches, key=lambda x: x.distance)
        used_points1, used_points2 = set(), set()
        filtered_matches = []

        for match in sorted_matches:
            pt1 = match.kp1.pt
            pt2 = match.kp2.pt

            is_duplicate = any(
                np.linalg.norm(np.array(pt1) - np.array(used_pt1)) < self.distance_threshold
                for used_pt1 in used_points1
            ) or any(
                np.linalg.norm(np.array(pt2) - np.array(used_pt2)) < self.distance_threshold
                for used_pt2 in used_points2
            )

            if not is_duplicate:
                filtered_matches.append(match)
                used_points1.add(pt1)
                used_points2.add(pt2)

        return filtered_matches

def create_mask(image_shape, points, depth_image=None, min_depth=0.3, max_depth=0.6):
    """Create a mask of ROI and depth"""
    roi_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [points], 255)
    
    if depth_image is None:
        return roi_mask
        
    depth_mask = np.zeros_like(roi_mask)
    valid_depth = (depth_image > min_depth * 1000) & (depth_image < max_depth * 1000)
    depth_mask[valid_depth] = 255
    
    return cv2.bitwise_and(roi_mask, depth_mask)

def load_roi_from_json(json_path):
    """Load ROI points from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['shapes'][0]['points'], dtype=np.int32)

def convert_matches_to_3d_points(matches, depth_image1, depth_image2):
    """Convert 2D matching points to 3D point pairs"""
    points_3d_1, points_3d_2 = [], []
    valid_matches = []
    
    for match in matches:
        u1, v1 = map(int, match.kp1.pt)
        u2, v2 = map(int, match.kp2.pt)
        
        if (0 <= u1 < depth_image1.shape[1] and 0 <= v1 < depth_image1.shape[0] and
            0 <= u2 < depth_image2.shape[1] and 0 <= v2 < depth_image2.shape[0]):
            
            z1 = depth_image1[v1, u1] / 1000.0
            z2 = depth_image2[v2, u2] / 1000.0
            
            if z1 > 0 and z2 > 0:
                x1 = (u1 - cx) * z1 / fx
                y1 = (v1 - cy) * z1 / fy
                x2 = (u2 - cx) * z2 / fx
                y2 = (v2 - cy) * z2 / fy
                
                points_3d_1.append([x1, y1, z1])
                points_3d_2.append([x2, y2, z2])
                valid_matches.append(match)

    return np.array(points_3d_1), np.array(points_3d_2), valid_matches

def compute_rigid_transform_svd(src_pts, dst_pts):
    """Using SVD to calculate rigid body transformation"""
    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)
    
    src_centered = src_pts - centroid_src
    dst_centered = dst_pts - centroid_dst
    
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_dst - R @ centroid_src
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def ransac_3d_rigid_transform(pts1, pts2, threshold=0.015, max_iters=1000):
    """Estimating 3D rigid transformation using RANSAC"""
    if len(pts1) < 3:
        return None, np.zeros(len(pts1), dtype=bool)

    best_inlier_count = 0
    best_inliers_mask = np.zeros(len(pts1), dtype=bool)
    best_trans = np.eye(4)

    for _ in range(max_iters):
        sample_idx = np.random.choice(len(pts1), 3, replace=False)
        T_est = compute_rigid_transform_svd(pts1[sample_idx], pts2[sample_idx])
        
        pts1_homo = np.hstack([pts1, np.ones((len(pts1), 1))])
        transformed = (T_est @ pts1_homo.T).T
        errors = np.linalg.norm(transformed[:, :3] - pts2, axis=1)
        
        inliers_mask = errors < threshold
        inlier_count = np.sum(inliers_mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers_mask = inliers_mask
            best_trans = T_est
            
            if inlier_count > 0.9 * len(pts1):
                break

    if best_inlier_count >= 3:
        refined_src = pts1[best_inliers_mask]
        refined_dst = pts2[best_inliers_mask]
        best_trans = compute_rigid_transform_svd(refined_src, refined_dst)

    return best_trans, best_inliers_mask

def compute_transformation_matrix(points_3d_1, points_3d_2):
    """Compute the transformation matrix between 3D point pairs"""
    if len(points_3d_1) < 3:
        return np.eye(4), 0.0

    T_est, inliers_mask = ransac_3d_rigid_transform(points_3d_1, points_3d_2)
    if np.sum(inliers_mask) < 3:
        return np.eye(4), 0.0

    pts1_homo = np.hstack([points_3d_1, np.ones((len(points_3d_1), 1))])
    transformed = (T_est @ pts1_homo.T).T
    errors = np.linalg.norm(transformed[:, :3] - points_3d_2, axis=1)
    
    final_inliers = errors < 0.015
    fitness = np.sum(final_inliers) / len(points_3d_1)
    
    return T_est, fitness

def select_object():
    """Select object to process"""
    print("\nAvailable objects:")
    for i, (key, config) in enumerate(OBJECT_CONFIGS.items(), 1):
        print(f"{i}. {config['name']} ({config['num_frames']} frames)")
    
    while True:
        try:
            choice = int(input("\nSelect object number (1-4): "))
            if 1 <= choice <= 4:
                return list(OBJECT_CONFIGS.keys())[choice - 1]
            print("Invalid choice. Please select a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_distance_threshold():
    """Function to let user adjust the distance threshold"""
    print("\nCurrent distance threshold for filtering duplicate matches is: 0")
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
        print("Using default distance threshold: 0")
        return 0

def main():
    # Select object to process
    object_key = select_object()
    object_config = OBJECT_CONFIGS[object_key]
    object_name = object_config['name']
    num_images = object_config['num_frames']
    
    # Select distance threshold
    distance_threshold = select_distance_threshold()
    
    print(f"\nProcessing {object_name} with {num_images} frames...")
    print(f"Using distance threshold: {distance_threshold}")
    
    matcher = FeatureMatcher(distance_threshold=distance_threshold)
    edge_info_list = []
    grayscale_images = []
    depth_images = []
    masks = []

    # Loading image and depth data
    print("Loading images...")
    for i in range(1, num_images + 1):
        gray_path = f'{object_name}/HSI/grayscale/image{i}.png'
        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            raise ValueError(f"Could not read grayscale image: {gray_path}")
        grayscale_images.append(gray_img)

        depth_path = f'{object_name}/HSI/Depth/transformed_undistorted_depth_{i}.png'
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise ValueError(f"Could not read depth image: {depth_path}")
        depth_images.append(depth_img)

        json_path = f'{object_name}/HSI/mask/undistorted_hsi_{i}.json'
        roi_points = load_roi_from_json(json_path)
        mask = create_mask(gray_img.shape[:2], roi_points, depth_img)
        masks.append(mask)

    # Processing adjacent frame pairs
    print("\nProcessing image pairs...")
    fallback_transformation = np.identity(4)

    for i in range(1, num_images + 1):
        next_i = 1 if i == num_images else i + 1
        curr_idx = i - 1
        next_idx = next_i - 1
        
        print(f"\nProcessing pair {i} -> {next_i}")

        # Feature matching and filtering
        raw_matches = matcher.match_features(
            grayscale_images[curr_idx], 
            grayscale_images[next_idx], 
            masks[curr_idx], 
            masks[next_idx]
        )
        filtered_matches = matcher.filter_duplicate_matches(raw_matches)
        
        if len(filtered_matches) < 3:
            transformation_matrix = fallback_transformation
            fitness = 0.0
            valid_3d_matches_count = 0
        else:
            # 3D point conversion and transformation matrix calculation
            points_3d_1, points_3d_2, valid_3d_matches = convert_matches_to_3d_points(
                filtered_matches, 
                depth_images[curr_idx], 
                depth_images[next_idx]
            )
            
            if len(valid_3d_matches) < 3:
                transformation_matrix = fallback_transformation
                fitness = 0.0
                valid_3d_matches_count = 0
            else:
                transformation_matrix, fitness = compute_transformation_matrix(
                    points_3d_1, points_3d_2
                )
                valid_3d_matches_count = len(valid_3d_matches)

        # Save the results
        edge_info = {
            'source': i,
            'target': next_i,
            'transformation': transformation_matrix,
            'fitness': fitness,
            'valid_matches_count': valid_3d_matches_count,
            'distance_threshold': distance_threshold
        }
        edge_info_list.append(edge_info)
        
        # Create output directory with object name if it doesn't exist
        output_dir = f'transformation/Mean/{object_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with object name in the path
        np.savez(f'{output_dir}/transformation_{i}_{next_i}.npz', 
                 transformation=transformation_matrix,
                 fitness=fitness,
                 valid_matches_count=valid_3d_matches_count,
                 distance_threshold=distance_threshold)

    print("\nAll transformations have been computed and saved.")
    print(f"Distance threshold used: {distance_threshold}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise