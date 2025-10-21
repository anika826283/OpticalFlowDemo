"""
XFeat neural network components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class XFeatBackbone(nn.Module):
    """
    Lightweight feature extraction backbone for XFeat
    """
    def __init__(self):
        super(XFeatBackbone, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Feature extraction at different scales
        x1 = self.layer1(x)    # 1/4 resolution
        x2 = self.layer2(x1)   # 1/8 resolution
        x3 = self.layer3(x2)   # 1/16 resolution

        return x3


class FeatureExtractor(nn.Module):
    """
    Feature extractor that outputs both detection scores and descriptors
    """
    def __init__(self, feature_dim=64):
        super(FeatureExtractor, self).__init__()

        self.backbone = XFeatBackbone()

        # Detection head (keypoint scores)
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 65, kernel_size=1)  # 64 bins + dustbin
        )

        # Descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            scores: [B, 65, H/16, W/16] detection scores
            descriptors: [B, D, H/16, W/16] dense descriptors
        """
        # Extract features
        features = self.backbone(x)

        # Detection scores
        scores = self.detection_head(features)

        # Descriptors
        descriptors = self.descriptor_head(features)

        # Normalize descriptors
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return scores, descriptors


class SimpleDetector(nn.Module):
    """
    Simple keypoint detector
    """
    def __init__(self, max_keypoints=4096, detection_threshold=0.005):
        super(SimpleDetector, self).__init__()
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold

    def forward(self, scores):
        """
        Extract keypoints from detection scores
        Args:
            scores: [B, 65, H, W] detection scores
        Returns:
            keypoints: [B, N, 2] keypoint locations (x, y)
            scores: [B, N] keypoint scores
        """
        b, c, h, w = scores.shape

        # Remove dustbin
        scores = scores[:, :-1, :, :]  # [B, 64, H, W]

        # Reshape to pixel-level scores
        # This is a simplified version - in reality XFeat uses sub-pixel refinement
        scores = scores.reshape(b, 8, 8, h, w)
        scores = scores.permute(0, 3, 1, 4, 2).reshape(b, h * 8, w * 8)

        # Apply threshold
        mask = scores > self.detection_threshold

        keypoints_list = []
        scores_list = []

        for i in range(b):
            # Get valid keypoints for this batch
            valid_y, valid_x = torch.where(mask[i])
            valid_scores = scores[i, valid_y, valid_x]

            # Limit number of keypoints
            if len(valid_scores) > self.max_keypoints:
                # Select top-k
                top_k_indices = torch.topk(valid_scores, self.max_keypoints).indices
                valid_x = valid_x[top_k_indices]
                valid_y = valid_y[top_k_indices]
                valid_scores = valid_scores[top_k_indices]

            # Stack as [N, 2]
            kpts = torch.stack([valid_x.float(), valid_y.float()], dim=1)

            keypoints_list.append(kpts)
            scores_list.append(valid_scores)

        return keypoints_list, scores_list


class MatcherHead(nn.Module):
    """
    Feature matcher using optimal transport / Sinkhorn algorithm
    """
    def __init__(self, feature_dim=64, bin_score=1.0):
        super(MatcherHead, self).__init__()
        self.bin_score = bin_score
        self.feature_dim = feature_dim

    def forward(self, descriptors0, descriptors1):
        """
        Match descriptors using similarity matrix
        Args:
            descriptors0: [B, N0, D]
            descriptors1: [B, N1, D]
        Returns:
            matches: list of [M, 2] match indices for each batch
            match_scores: list of [M] match scores
        """
        b = descriptors0.shape[0]

        matches_list = []
        scores_list = []

        for i in range(b):
            desc0 = descriptors0[i]  # [N0, D]
            desc1 = descriptors1[i]  # [N1, D]

            # Compute similarity matrix
            sim = torch.matmul(desc0, desc1.t())  # [N0, N1]

            # Apply softmax on both dimensions for mutual nearest neighbor
            sim0 = F.softmax(sim, dim=1)
            sim1 = F.softmax(sim, dim=0)

            # Mutual nearest neighbor
            mutual = sim0 * sim1

            # Get top matches
            max_scores0, max_idx0 = mutual.max(dim=1)
            max_scores1, max_idx1 = mutual.max(dim=0)

            # Mutual check
            mutual_check = (max_idx1[max_idx0] == torch.arange(len(desc0), device=desc0.device))

            # Valid matches
            valid_indices = torch.where(mutual_check)[0]
            matched_indices1 = max_idx0[valid_indices]

            if len(valid_indices) > 0:
                matches = torch.stack([valid_indices, matched_indices1], dim=1)
                match_scores = max_scores0[valid_indices]
            else:
                matches = torch.zeros((0, 2), dtype=torch.long, device=desc0.device)
                match_scores = torch.zeros(0, device=desc0.device)

            matches_list.append(matches)
            scores_list.append(match_scores)

        return matches_list, scores_list
