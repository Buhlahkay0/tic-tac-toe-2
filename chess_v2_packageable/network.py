import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import PLANES, OUTPUT_DIM


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _pick_device()
print("Using device:", device)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, channels=128, blocks=8, output_dim=OUTPUT_DIM):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.input_block = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)]
        )
        # Policy head: 2-filter conv → flatten → linear
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, output_dim),
        )
        # Value head: 1-filter conv → flatten → FC(256) → FC(1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_blocks(x)
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return policy, value


def load_checkpoint(path, map_location=None):
    """Load a checkpoint dict and build a ChessNet with the stored architecture."""
    data = torch.load(path, map_location=map_location or device)
    channels = data.get("channels", 128)
    blocks = data.get("blocks", 8)
    net = ChessNet(channels=channels, blocks=blocks).to(map_location or device)
    state = data["model_state_dict"] if "model_state_dict" in data else data
    net.load_state_dict(state)
    return net, data


if __name__ == "__main__":
    net = ChessNet().to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
