"""
Transformer module for GMFlow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PositionEncodingSine(nn.Module):
    """
    Position encoding with sine and cosine functions
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.size()

        y_embed = torch.arange(h, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device)

        y_embed = y_embed.view(h, 1).repeat(1, w)
        x_embed = x_embed.view(1, w).repeat(h, 1)

        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * dim_t / self.d_model)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)

        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)  # [d_model, h, w]
        pos = pos.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, d_model, h, w]

        return pos


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, d_model=128, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, L, C]
        b, l, c = x.shape

        q = self.q_proj(x).view(b, l, self.num_heads, self.d_head)
        k = self.k_proj(x).view(b, l, self.num_heads, self.d_head)
        v = self.v_proj(x).view(b, l, self.num_heads, self.d_head)

        # Transpose for attention computation: [B, num_heads, L, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, num_heads, L, d_head]
        out = out.transpose(1, 2).contiguous().view(b, l, c)  # [B, L, C]

        out = self.merge(out)

        return out


class TransformerLayer(nn.Module):
    """Single transformer layer"""
    def __init__(self, d_model=128, num_heads=1, ffn_dim_expansion=4):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_dim_expansion),
            nn.GELU(),
            nn.Linear(d_model * ffn_dim_expansion, d_model),
        )

    def forward(self, x):
        # x: [B, L, C]
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))

        # FFN with residual connection
        x = x + self.ffn(self.norm2(x))

        return x


class FeatureTransformer(nn.Module):
    """Feature Transformer for GMFlow"""
    def __init__(self, num_layers=6, d_model=128, num_heads=1, ffn_dim_expansion=4):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, ffn_dim_expansion)
            for _ in range(num_layers)
        ])

        self.pos_encoding = PositionEncodingSine(d_model)

    def forward(self, feature0, feature1):
        """
        Args:
            feature0: [B, C, H, W]
            feature1: [B, C, H, W]
        Returns:
            feature0_new: [B, C, H, W]
            feature1_new: [B, C, H, W]
        """
        b, c, h, w = feature0.shape

        # Add positional encoding
        pos_enc = self.pos_encoding(feature0)

        feature0 = feature0 + pos_enc
        feature1 = feature1 + pos_enc

        # Reshape for transformer: [B, C, H, W] -> [B, H*W, C]
        feature0 = rearrange(feature0, 'b c h w -> b (h w) c')
        feature1 = rearrange(feature1, 'b c h w -> b (h w) c')

        # Concatenate features from both images
        concat_feature = torch.cat([feature0, feature1], dim=1)  # [B, 2*H*W, C]

        # Apply transformer layers
        for layer in self.layers:
            concat_feature = layer(concat_feature)

        # Split features
        feature0_new, feature1_new = torch.split(concat_feature, [h * w, h * w], dim=1)

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        feature0_new = rearrange(feature0_new, 'b (h w) c -> b c h w', h=h, w=w)
        feature1_new = rearrange(feature1_new, 'b (h w) c -> b c h w', h=h, w=w)

        return feature0_new, feature1_new
