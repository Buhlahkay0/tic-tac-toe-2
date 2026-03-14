import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

OUTPUT_DIM = 4672


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
    def __init__(self, num_res_blocks=5, channels=256, output_dim=OUTPUT_DIM):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
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


_WHITE_CH = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
             chess.ROOK: 3, chess.QUEEN:  4, chess.KING:   5}
_BLACK_CH = {chess.PAWN: 6, chess.KNIGHT: 7, chess.BISHOP: 8,
             chess.ROOK: 9, chess.QUEEN: 10, chess.KING:  11}


def board_to_tensor(chess_board, device=device):
    """
    Convert a python-chess Board into a 12x8x8 tensor on the specified device.

    Channels:
      0: White Pawn,   1: White Knight,  2: White Bishop,
      3: White Rook,   4: White Queen,   5: White King,
      6: Black Pawn,   7: Black Knight,  8: Black Bishop,
      9: Black Rook,  10: Black Queen,  11: Black King.

    Row 0 corresponds to rank 8.
    """
    arr = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in chess_board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        ch = _WHITE_CH[piece.piece_type] if piece.color == chess.WHITE else _BLACK_CH[piece.piece_type]
        arr[ch, row, col] = 1.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def boards_to_batch(boards):
    """
    Convert a list of chess.Board objects into a single (N, 12, 8, 8) CPU tensor.
    Caller does one .to(device) for the whole batch instead of N individual transfers.
    """
    arr = np.zeros((len(boards), 12, 8, 8), dtype=np.float32)
    for i, board in enumerate(boards):
        for square, piece in board.piece_map().items():
            row = 7 - (square // 8)
            col = square % 8
            ch = _WHITE_CH[piece.piece_type] if piece.color == chess.WHITE else _BLACK_CH[piece.piece_type]
            arr[i, ch, row, col] = 1.0
    return torch.from_numpy(arr)


if __name__ == "__main__":
    net = ChessNet().to(device)
    print("Network loaded on device:", next(net.parameters()).device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
