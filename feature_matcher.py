# feature_matcher.py
import cv2
import numpy as np
from config import CommonConfig, MatchInfo

class FeatureMatcher:
    """Feature extraction and matching class"""

    def __init__(self, distance_threshold=None):
        """Initialize feature matcher"""
        params = CommonConfig.FEATURE_PARAMS
        self.sift = cv2.SIFT_create(
            nfeatures=params['nfeatures'],
            contrastThreshold=params['contrastThreshold'],
            edgeThreshold=params['edgeThreshold'],
        )
        self.matcher = cv2.BFMatcher()
        self.distance_threshold = distance_threshold if distance_threshold is not None else params['default_distance_threshold']
        self.ratio_threshold = params['ratio_threshold']
        self.min_matches = params['min_matches']

    def match_single_channel(self, img1_channel, img2_channel, mask1=None, mask2=None):
        """Perform feature extraction and matching on a single channel"""
        kp1, des1 = self.sift.detectAndCompute(img1_channel, mask1)
        kp2, des2 = self.sift.detectAndCompute(img2_channel, mask2)

        if des1 is None or des2 is None or len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return []

        matches = self.matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(MatchInfo(kp1[m.queryIdx], kp2[m.trainIdx], m.distance))
        return good_matches

    def match_features(self, img_bands1, img_bands2, mask1=None, mask2=None):
        """Perform feature matching on all provided bands and merge results"""
        all_matches = []
        for band1, band2 in zip(img_bands1, img_bands2):
            matches = self.match_single_channel(band1, band2, mask1, mask2)
            all_matches.extend(matches)
        return all_matches

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
            if not is_duplicate:
                filtered_matches.append(match)
                used_points1.add(pt1)
                used_points2.add(pt2)

        return filtered_matches