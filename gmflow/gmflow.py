"""
GMFlow: Learning Optical Flow via Global Matching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .geometry import coords_grid, bilinear_sample


class GMFlow(nn.Module):
    """
    GMFlow model for optical flow estimation
    """
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_transformer_layers=6,
                 num_head=1,
                 ffn_dim_expansion=4,
                 attn_splits_list=[2],
                 corr_radius_list=[-1],
                 prop_radius_list=[-1],
                 reg_refine=False,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN backbone for feature extraction
        self.backbone = CNNEncoder(output_dim=feature_channels, norm_fn='batch')

        # Transformer for feature enhancement
        self.transformer = FeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            num_heads=num_head,
            ffn_dim_expansion=ffn_dim_expansion
        )

        # Flow prediction head
        self.flow_pred_head = nn.Sequential(
            nn.Conv2d(feature_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 3, padding=1)
        )

    def extract_feature(self, img):
        """
        Extract features from image
        Args:
            img: [B, 3, H, W]
        Returns:
            feature: [B, C, H//8, W//8]
        """
        # Normalize to [0, 1]
        concat = torch.cat([img], dim=0)
        features = self.backbone(concat)  # [B, C, H//8, W//8]

        return features

    def global_correlation_softmax(self, feature0, feature1):
        """
        Global correlation with softmax for matching
        Args:
            feature0: [B, C, H, W]
            feature1: [B, C, H, W]
        Returns:
            flow: [B, 2, H, W]
        """
        b, c, h, w = feature0.shape

        # Normalize features
        feature0 = F.normalize(feature0, p=2, dim=1)
        feature1 = F.normalize(feature1, p=2, dim=1)

        # Reshape for correlation: [B, C, H, W] -> [B, H*W, C]
        feature0_flat = rearrange(feature0, 'b c h w -> b (h w) c')
        feature1_flat = rearrange(feature1, 'b c h w -> b (h w) c')

        # Compute correlation: [B, H*W, H*W]
        correlation = torch.matmul(feature0_flat, feature1_flat.transpose(-2, -1))
        correlation = correlation / (c ** 0.5)

        # Apply softmax
        prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

        # Generate coordinate grid for feature1
        coords = coords_grid(b, h, w, device=feature0.device)  # [B, 2, H, W]
        coords_flat = rearrange(coords, 'b c h w -> b (h w) c')  # [B, H*W, 2]

        # Compute expected coordinates using probability
        coords_warped = torch.matmul(prob, coords_flat)  # [B, H*W, 2]
        coords_warped = rearrange(coords_warped, 'b (h w) c -> b c h w', h=h, w=w)  # [B, 2, H, W]

        # Flow is the difference between warped coordinates and original coordinates
        flow = coords_warped - coords  # [B, 2, H, W]

        return flow

    def upsample_flow(self, flow, mask=None, factor=8):
        """
        Upsample flow field
        Args:
            flow: [B, 2, H, W]
            mask: optional mask for convex upsampling
            factor: upsampling factor
        Returns:
            upsampled_flow: [B, 2, H*factor, W*factor]
        """
        b, c, h, w = flow.shape
        new_h, new_w = h * factor, w * factor

        # Simple bilinear upsampling
        flow = F.interpolate(flow, size=(new_h, new_w), mode='bilinear', align_corners=True)
        flow = flow * factor  # Scale flow values

        return flow

    def forward(self, img0, img1, attn_splits_list=None, corr_radius_list=None, prop_radius_list=None):
        """
        Forward pass
        Args:
            img0: [B, 3, H, W]
            img1: [B, 3, H, W]
        Returns:
            results_dict: dictionary containing:
                - flow: [B, 2, H, W]
        """
        results_dict = {}

        # Extract features
        feature0 = self.extract_feature(img0)  # [B, C, H//8, W//8]
        feature1 = self.extract_feature(img1)  # [B, C, H//8, W//8]

        # Enhance features with transformer
        feature0_enhanced, feature1_enhanced = self.transformer(feature0, feature1)

        # Global correlation and matching
        flow_low = self.global_correlation_softmax(feature0_enhanced, feature1_enhanced)

        # Upsample flow to original resolution
        flow = self.upsample_flow(flow_low, factor=self.upsample_factor)

        results_dict['flow'] = flow

        return results_dict

    @torch.no_grad()
    def inference(self, img0, img1):
        """
        Inference mode
        Args:
            img0: [B, 3, H, W]
            img1: [B, 3, H, W]
        Returns:
            flow: [B, 2, H, W]
        """
        results = self.forward(img0, img1)
        return results['flow']
