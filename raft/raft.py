"""
RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .update import BasicUpdateBlock, SmallUpdateBlock
from .utils import coords_grid, upflow8


class RAFT(nn.Module):
    """
    RAFT optical flow model
    """
    def __init__(self,
                 small=False,
                 dropout=0.0,
                 alternate_corr=False,
                 mixed_precision=False):
        """
        Args:
            small: use small model for faster inference
            dropout: dropout rate
            alternate_corr: use alternate correlation implementation
            mixed_precision: use mixed precision training
        """
        super(RAFT, self).__init__()

        self.small = small
        self.alternate_corr = alternate_corr
        self.mixed_precision = mixed_precision

        if small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4

        # Feature encoder
        if small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=dropout)
            self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=dropout)
            self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)

    def freeze_bn(self):
        """Freeze batch normalization layers"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """
        Initialize flow as zero
        Args:
            img: input image [B, 3, H, W]
        Returns:
            flow coordinates [B, 2, H//8, W//8]
        """
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """
        Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """
        Estimate optical flow between image1 and image2
        Args:
            image1: first image [B, 3, H, W]
            image2: second image [B, 3, H, W]
            iters: number of update iterations
            flow_init: optional initial flow
            test_mode: if True, return only final flow
        Returns:
            flow_predictions: list of flow predictions at each iteration
        """
        # Normalize images to [0, 1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # Run the feature network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            fmap1 = self.fnet(image1)
            fmap2 = self.fnet(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # Build correlation pyramid
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # Run the context network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize flow
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # Index correlation volume

            flow = coords1 - coords0
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                net, delta_flow = self.update_block(net, inp, corr, flow)

            # Update flow
            coords1 = coords1 + delta_flow

            # Upsample flow to full resolution
            if itr < iters - 1 or not test_mode:
                flow_up = upflow8(coords1 - coords0)
                flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions

    @torch.no_grad()
    def inference(self, image1, image2, iters=12):
        """
        Inference mode
        Args:
            image1: first image [B, 3, H, W]
            image2: second image [B, 3, H, W]
            iters: number of update iterations
        Returns:
            flow: optical flow [B, 2, H, W]
        """
        _, flow_up = self.forward(image1, image2, iters=iters, test_mode=True)
        return flow_up
