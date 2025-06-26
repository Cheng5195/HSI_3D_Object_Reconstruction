# transformation_estimator.py
import numpy as np
from config import CommonConfig

class TransformationEstimator:
    """3D transformation estimation class"""

    @staticmethod
    def convert_to_3d_points(matches, depth_image1, depth_image2):
        """Convert 2D matching points to 3D point pairs"""
        fx = CommonConfig.CAMERA_PARAMS['fx']
        fy = CommonConfig.CAMERA_PARAMS['fy']
        cx = CommonConfig.CAMERA_PARAMS['cx']
        cy = CommonConfig.CAMERA_PARAMS['cy']
        scale_factor = CommonConfig.DEPTH_PARAMS['scale_factor']

        points_3d_1, points_3d_2 = [], []
        valid_matches = []

        for match in matches:
            u1, v1 = map(int, match.kp1.pt)
            u2, v2 = map(int, match.kp2.pt)

            if (0 <= u1 < depth_image1.shape[1] and 0 <= v1 < depth_image1.shape[0] and
                0 <= u2 < depth_image2.shape[1] and 0 <= v2 < depth_image2.shape[0]):
                z1 = depth_image1[v1, u1] / scale_factor
                z2 = depth_image2[v2, u2] / scale_factor

                if z1 > 0 and z2 > 0:
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

    @classmethod
    def ransac_3d_rigid_transform(cls, pts1, pts2, threshold=None, max_iters=None):
        """Estimate 3D rigid transformation using RANSAC"""
        if threshold is None: threshold = CommonConfig.RANSAC_PARAMS['threshold']
        if max_iters is None: max_iters = CommonConfig.RANSAC_PARAMS['max_iters']
        min_inliers = CommonConfig.RANSAC_PARAMS['min_inliers']
        success_ratio = CommonConfig.RANSAC_PARAMS['success_ratio']

        if len(pts1) < min_inliers:
            return None, np.zeros(len(pts1), dtype=bool)

        best_inlier_count = 0
        best_inliers_mask = np.zeros(len(pts1), dtype=bool)
        best_trans = np.eye(4)

        for _ in range(max_iters):
            sample_idx = np.random.choice(len(pts1), min_inliers, replace=False)
            T_est = cls.compute_rigid_transform_svd(pts1[sample_idx], pts2[sample_idx])
            pts1_homo = np.hstack([pts1, np.ones((len(pts1), 1))])
            transformed = (T_est @ pts1_homo.T).T
            errors = np.linalg.norm(transformed[:, :3] - pts2, axis=1)
            inliers_mask = errors < threshold
            inlier_count = np.sum(inliers_mask)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers_mask = inliers_mask
                best_trans = T_est
                if inlier_count > success_ratio * len(pts1):
                    break

        if best_inlier_count >= min_inliers:
            refined_src = pts1[best_inliers_mask]
            refined_dst = pts2[best_inliers_mask]
            best_trans = cls.compute_rigid_transform_svd(refined_src, refined_dst)

        return best_trans, best_inliers_mask

    @classmethod
    def compute_transformation_matrix(cls, points_3d_1, points_3d_2, threshold=None):
        """Compute transformation matrix between 3D point pairs"""
        min_inliers = CommonConfig.RANSAC_PARAMS['min_inliers']
        if len(points_3d_1) < min_inliers:
            return np.eye(4), 0.0

        T_est, inliers_mask = cls.ransac_3d_rigid_transform(points_3d_1, points_3d_2, threshold)
        if np.sum(inliers_mask) < min_inliers:
            return np.eye(4), 0.0

        pts1_homo = np.hstack([points_3d_1, np.ones((len(points_3d_1), 1))])
        transformed = (T_est @ pts1_homo.T).T
        if threshold is None: threshold = CommonConfig.RANSAC_PARAMS['threshold']
        errors = np.linalg.norm(transformed[:, :3] - points_3d_2, axis=1)
        final_inliers = errors < threshold
        fitness = np.sum(final_inliers) / len(points_3d_1) if len(points_3d_1) > 0 else 0.0

        return T_est, fitness