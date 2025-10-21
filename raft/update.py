"""
Update operator for RAFT - GRU-based iterative refinement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    """Predict flow from hidden state"""
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    """Convolutional GRU cell"""
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        """
        Args:
            h: hidden state [B, hidden_dim, H, W]
            x: input [B, input_dim, H, W]
        Returns:
            h_new: new hidden state [B, hidden_dim, H, W]
        """
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    """Separable Convolutional GRU for efficiency"""
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # Horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # Vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    """
    Small motion encoder
    Encodes correlation and flow into a feature
    """
    def __init__(self, corr_levels=4, corr_radius=4):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        """
        Args:
            flow: current flow estimate [B, 2, H, W]
            corr: correlation features [B, C, H, W]
        Returns:
            motion features [B, 80, H, W]
        """
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    """
    Basic motion encoder
    Encodes correlation and flow into a feature
    """
    def __init__(self, corr_levels=4, corr_radius=4):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(256, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        """
        Args:
            flow: current flow estimate [B, 2, H, W]
            corr: correlation features [B, C, H, W]
        Returns:
            motion features [B, 128, H, W]
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    """
    Basic update block for RAFT
    """
    def __init__(self, corr_levels=4, corr_radius=4, hidden_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, corr, flow):
        """
        Args:
            net: hidden state [B, hidden_dim, H, W]
            inp: context features [B, hidden_dim, H, W]
            corr: correlation features [B, C, H, W]
            flow: current flow estimate [B, 2, H, W]
        Returns:
            net: updated hidden state
            delta_flow: flow update
        """
        motion_features = self.encoder(flow, corr)
        inp_cat = torch.cat([inp, motion_features], dim=1)

        # Update hidden state
        net = self.gru(net, inp_cat)

        # Predict flow update
        delta_flow = self.flow_head(net)

        return net, delta_flow


class SmallUpdateBlock(nn.Module):
    """
    Small update block for faster inference
    """
    def __init__(self, corr_levels=4, corr_radius=4, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(corr_levels, corr_radius)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp_cat = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)

        return net, delta_flow
