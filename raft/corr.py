"""
Correlation volume computation for RAFT
"""
import torch
import torch.nn.functional as F


class CorrBlock:
    """
    Correlation block that builds a 4D correlation pyramid
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        Args:
            fmap1: feature map from image 1, [B, C, H, W]
            fmap2: feature map from image 2, [B, C, H, W]
            num_levels: number of pyramid levels
            radius: correlation lookup radius
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Compute correlation volume
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        # Compute correlation
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)  # [B, H*W, H*W]
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr / torch.sqrt(torch.tensor(dim).float())

        # Build correlation pyramid
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr.view(batch * ht * wd, 1, ht, wd), 2, stride=2)
            corr = corr.view(batch, ht, wd, 1, corr.shape[2], corr.shape[3])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """
        Lookup correlation values at given coordinates
        Args:
            coords: coordinates, [B, 2, H, W]
        Returns:
            corr_features: correlation features, [B, num_levels*(2*r+1)^2, H, W]
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            _, _, _, _, h2, w2 = corr.shape

            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  # [B, H, W, C]
        return out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

    def bilinear_sampler(self, corr, coords):
        """
        Bilinear sampling of correlation volume
        Args:
            corr: correlation volume, [B, H1, W1, 1, H2, W2]
            coords: sampling coordinates, [B*H1*W1, 2*r+1, 2*r+1, 2]
        Returns:
            sampled values
        """
        batch, h1, w1, dim, h2, w2 = corr.shape

        # Normalize coordinates to [-1, 1]
        coords_x = 2 * coords[..., 0] / (w2 - 1) - 1
        coords_y = 2 * coords[..., 1] / (h2 - 1) - 1
        coords_norm = torch.stack([coords_x, coords_y], dim=-1)

        # Reshape for grid_sample
        corr = corr.view(batch * h1 * w1, dim, h2, w2)
        coords_norm = coords_norm.view(batch * h1 * w1, -1, 1, 2)

        # Sample
        sampled = F.grid_sample(corr, coords_norm, align_corners=True, mode='bilinear')
        sampled = sampled.view(batch, h1, w1, -1)

        return sampled


class AlternateCorrBlock:
    """
    Alternative correlation block with different sampling strategy
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        # Normalize features
        fmap1 = fmap1 / torch.sqrt(torch.sum(fmap1 ** 2, dim=1, keepdim=True))
        fmap2 = fmap2 / torch.sqrt(torch.sum(fmap2 ** 2, dim=1, keepdim=True))

        # Compute correlation
        self.corr = torch.matmul(fmap1.transpose(1, 2), fmap2)  # [B, H*W, H*W]
        self.corr = self.corr.view(batch, ht, wd, 1, ht, wd)

        # Build pyramid
        self.corr_pyramid = [self.corr]
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(self.corr.view(batch * ht * wd, 1, ht, wd), 2, stride=2)
            self.corr_pyramid.append(corr.view(batch, ht, wd, 1, corr.shape[2], corr.shape[3]))

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            _, _, _, _, h2, w2 = corr.shape
            corr_sampled = self.bilinear_sampler(corr, coords_lvl, h2, w2)
            corr_sampled = corr_sampled.view(batch, h1, w1, -1)
            out_pyramid.append(corr_sampled)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    def bilinear_sampler(self, corr, coords, h2, w2):
        batch, h1, w1, dim = corr.shape[:4]

        coords_x = 2 * coords[..., 0] / (w2 - 1) - 1
        coords_y = 2 * coords[..., 1] / (h2 - 1) - 1
        coords_norm = torch.stack([coords_x, coords_y], dim=-1)

        corr = corr.view(batch * h1 * w1, dim, h2, w2)
        coords_norm = coords_norm.view(batch * h1 * w1, -1, 1, 2)

        sampled = F.grid_sample(corr, coords_norm, align_corners=True, mode='bilinear')
        return sampled.view(batch, h1, w1, -1)
