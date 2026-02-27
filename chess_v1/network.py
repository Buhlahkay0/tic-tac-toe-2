import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

# Define the fixed output dimension.
# (A production implementation would design a complete move encoding covering all moves.)
OUTPUT_DIM = 4672

class ChessNet(nn.Module):
    def __init__(self, output_dim=OUTPUT_DIM):
        super(ChessNet, self).__init__()
        # The board is represented as a 12-channel 8x8 tensor:
        # Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        # Channels 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.policy_head = nn.Linear(1024, output_dim)
        self.value_head = nn.Linear(1024, 1)

    def forward(self, x):
        # x shape: [batch_size, 12, 8, 8]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        policy = self.policy_head(x)
        policy = F.log_softmax(policy, dim=1)
        value = self.value_head(x)
        value = torch.tanh(value)
        return policy, value

def board_to_tensor(chess_board):
    """
    Convert a python-chess Board into a 12x8x8 tensor.
    
    Channels:
      0: White Pawn,   1: White Knight,  2: White Bishop,
      3: White Rook,   4: White Queen,   5: White King,
      6: Black Pawn,   7: Black Knight,  8: Black Bishop,
      9: Black Rook,  10: Black Queen,  11: Black King.
    
    The board is arranged so that row 0 corresponds to rank 8.
    """
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = chess_board.piece_map()  # {square: Piece}
    for square, piece in piece_map.items():
        row = 7 - (square // 8)   # Invert row so that 0 is rank 8.
        col = square % 8
        if piece.color == chess.WHITE:
            channel = {
                chess.PAWN:   0,
                chess.KNIGHT: 1,
                chess.BISHOP: 2,
                chess.ROOK:   3,
                chess.QUEEN:  4,
                chess.KING:   5
            }[piece.piece_type]
        else:
            channel = {
                chess.PAWN:   6,
                chess.KNIGHT: 7,
                chess.BISHOP: 8,
                chess.ROOK:   9,
                chess.QUEEN: 10,
                chess.KING:  11
            }[piece.piece_type]
        board_tensor[channel, row, col] = 1.0
    # Convert to a torch tensor and add a batch dimension.
    return torch.from_numpy(board_tensor).unsqueeze(0)
