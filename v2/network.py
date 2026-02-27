import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        # Two convolutional layers.
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Fully connected layer.
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        # Policy head: outputs log probabilities for 9 moves.
        self.policy_head = nn.Linear(128, 9)
        # Value head: outputs a scalar in [-1, 1].
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten the convolutional output.
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # Compute policy and value outputs.
        policy = self.policy_head(x)
        policy = F.log_softmax(policy, dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

def board_to_tensor(board, current_player):
    """
    Convert the board (3x3 numpy array) into a 2-channel tensor.
    Channel 0: positions where board equals current_player.
    Channel 1: positions where board equals -current_player.
    """
    board_tensor = np.zeros((2, 3, 3), dtype=np.float32)
    board_tensor[0] = (board == current_player).astype(np.float32)
    board_tensor[1] = (board == -current_player).astype(np.float32)
    return torch.from_numpy(board_tensor).unsqueeze(0)  # add batch dimension
