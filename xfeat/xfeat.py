"""
XFeat: Accelerated Features for Lightweight Image Matching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .net import FeatureExtractor, SimpleDetector, MatcherHead
from .interpolator import interpolate_dense


class XFeat(nn.Module):
    """
    XFeat model for feature detection, description, and matching
    """
    def __init__(self,
                 feature_dim=64,
                 max_keypoints=4096,
                 detection_threshold=0.005):
        """
        Args:
            feature_dim: dimension of feature descriptors
            max_keypoints: maximum number of keypoints to detect
            detection_threshold: threshold for keypoint detection
        """
        super(XFeat, self).__init__()

        self.feature_dim = feature_dim
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold

        # Feature extraction
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)

        # Keypoint detection
        self.detector = SimpleDetector(
            max_keypoints=max_keypoints,
            detection_threshold=detection_threshold
        )

        # Matcher
        self.matcher = MatcherHead(feature_dim=feature_dim)

    def extract_features(self, image):
        """
        Extract features from image
        Args:
            image: [B, 3, H, W] input image in range [0, 1]
        Returns:
            keypoints: list of [N, 2] keypoint locations for each batch
            descriptors: list of [N, D] descriptors for each batch
            scores: list of [N] keypoint scores for each batch
        """
        b, c, h, w = image.shape

        # Extract dense features
        detection_scores, dense_descriptors = self.feature_extractor(image)

        # Detect keypoints
        keypoints_list, scores_list = self.detector(detection_scores)

        # Sample descriptors at keypoint locations
        descriptors_list = []

        for i in range(b):
            if len(keypoints_list[i]) > 0:
                # Scale keypoints to descriptor resolution
                kpts_scaled = keypoints_list[i].clone()
                kpts_scaled[:, 0] = kpts_scaled[:, 0] * (dense_descriptors.shape[3] / (w * 8))
                kpts_scaled[:, 1] = kpts_scaled[:, 1] * (dense_descriptors.shape[2] / (h * 8))

                # Sample descriptors
                kpts_batch = kpts_scaled.unsqueeze(0)  # [1, N, 2]
                desc_batch = dense_descriptors[i:i+1]  # [1, D, H, W]

                descriptors = interpolate_dense(kpts_batch, desc_batch)  # [1, N, D]
                descriptors = descriptors[0]  # [N, D]
            else:
                descriptors = torch.zeros((0, self.feature_dim), device=image.device)

            descriptors_list.append(descriptors)

        return keypoints_list, descriptors_list, scores_list

    def match_features(self, descriptors0, descriptors1):
        """
        Match features between two images
        Args:
            descriptors0: list of [N0, D] descriptors
            descriptors1: list of [N1, D] descriptors
        Returns:
            matches: list of [M, 2] match indices
            match_scores: list of [M] confidence scores
        """
        # Pad descriptors to same length for batching
        max_len0 = max([d.shape[0] for d in descriptors0]) if descriptors0 else 0
        max_len1 = max([d.shape[0] for d in descriptors1]) if descriptors1 else 0

        if max_len0 == 0 or max_len1 == 0:
            return [torch.zeros((0, 2), dtype=torch.long) for _ in range(len(descriptors0))], \
                   [torch.zeros(0) for _ in range(len(descriptors0))]

        # Pad and stack
        batch_size = len(descriptors0)
        desc0_padded = torch.zeros(batch_size, max_len0, self.feature_dim, device=descriptors0[0].device)
        desc1_padded = torch.zeros(batch_size, max_len1, self.feature_dim, device=descriptors1[0].device)

        for i in range(batch_size):
            n0 = descriptors0[i].shape[0]
            n1 = descriptors1[i].shape[0]
            desc0_padded[i, :n0] = descriptors0[i]
            desc1_padded[i, :n1] = descriptors1[i]

        # Match
        matches, match_scores = self.matcher(desc0_padded, desc1_padded)

        return matches, match_scores

    def forward(self, image0, image1):
        """
        Full pipeline: extract and match features
        Args:
            image0: [B, 3, H, W] first image
            image1: [B, 3, H, W] second image
        Returns:
            dict containing:
                - keypoints0, keypoints1: detected keypoints
                - descriptors0, descriptors1: feature descriptors
                - matches: match indices
                - match_scores: match confidence scores
        """
        # Extract features
        kpts0, desc0, scores0 = self.extract_features(image0)
        kpts1, desc1, scores1 = self.extract_features(image1)

        # Match features
        matches, match_scores = self.match_features(desc0, desc1)

        return {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'matches': matches,
            'match_scores': match_scores
        }

    @torch.no_grad()
    def inference(self, image0, image1):
        """
        Inference mode
        Args:
            image0: [B, 3, H, W] first image in range [0, 255]
            image1: [B, 3, H, W] second image in range [0, 255]
        Returns:
            dict with keypoints, descriptors, and matches
        """
        # Normalize to [0, 1]
        image0 = image0 / 255.0
        image1 = image1 / 255.0

        return self.forward(image0, image1)

    def compute_sparse_flow(self, matches_dict, image_shape):
        """
        Compute sparse optical flow from matches
        Args:
            matches_dict: output from forward/inference
            image_shape: (H, W) original image shape
        Returns:
            sparse_flow: dict containing:
                - keypoints: [N, 2] matched keypoint locations in image0
                - flow: [N, 2] flow vectors (dx, dy)
                - confidence: [N] match confidence scores
        """
        batch_results = []

        keypoints0 = matches_dict['keypoints0']
        keypoints1 = matches_dict['keypoints1']
        matches = matches_dict['matches']
        match_scores = matches_dict['match_scores']

        for i in range(len(matches)):
            if len(matches[i]) > 0:
                # Get matched keypoints
                match_idx = matches[i]  # [M, 2]
                kpts0_matched = keypoints0[i][match_idx[:, 0]]  # [M, 2]
                kpts1_matched = keypoints1[i][match_idx[:, 1]]  # [M, 2]

                # Compute flow vectors
                flow_vectors = kpts1_matched - kpts0_matched  # [M, 2]

                batch_results.append({
                    'keypoints': kpts0_matched,
                    'flow': flow_vectors,
                    'confidence': match_scores[i]
                })
            else:
                batch_results.append({
                    'keypoints': torch.zeros((0, 2), device=keypoints0[0].device),
                    'flow': torch.zeros((0, 2), device=keypoints0[0].device),
                    'confidence': torch.zeros(0, device=keypoints0[0].device)
                })

        return batch_results
