import numpy as np
import torch
import torch.nn.functional as F
from mcts import MCTS
from tictactoe import TicTacToe
from network import board_to_tensor

def self_play_game(net, num_simulations=50):
    game = TicTacToe()
    mcts = MCTS(net, num_simulations=num_simulations)
    states, mcts_probs, current_players = [], [], []
    
    while not game.is_terminal():
        # Run MCTS to get the root node.
        root = mcts.search(game)
        # Compute the move probabilities from the visit counts.
        visit_counts = np.array([
            root.children[(i, j)].visit_count if (i, j) in root.children else 0 
            for i in range(3) for j in range(3)
        ])
        visit_sum = np.sum(visit_counts)
        if visit_sum > 0:
            probs = visit_counts / visit_sum
        else:
            probs = np.zeros(9)
        
        states.append(game.board.copy())
        mcts_probs.append(probs)
        current_players.append(game.current_player)
        
        # Select the move with the highest probability.
        move_index = np.argmax(probs)
        move = (move_index // 3, move_index % 3)
        game.make_move(move)
    
    winner = game.check_winner()
    # Assign rewards for each state.
    rewards = []
    for player in current_players:
        if winner == 0:
            rewards.append(0)
        else:
            reward = 1 if winner == player else -1
            rewards.append(reward)
    return states, mcts_probs, rewards

def train_network(net, optimizer, states, mcts_probs, rewards, epochs=1):
    net.train()
    for epoch in range(epochs):
        total_loss = 0
        for board, pi, reward in zip(states, mcts_probs, rewards):
            # Convert board state to tensor. The board is represented
            # from the perspective of the current player in self-play.
            board_tensor = board_to_tensor(board, 1)
            optimizer.zero_grad()
            log_pi, value = net(board_tensor)
            # Value loss: mean squared error between predicted value and reward.
            value_loss = F.mse_loss(value, torch.tensor([[reward]], dtype=torch.float32))
            # Policy loss: negative log likelihood.
            target_pi = torch.tensor(pi, dtype=torch.float32).unsqueeze(0)
            policy_loss = -torch.sum(target_pi * log_pi)
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(states):.4f}")
