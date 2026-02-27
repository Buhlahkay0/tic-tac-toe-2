from network import TicTacToeNet
from train import self_play_game, train_network
import torch

if __name__ == "__main__":
    net = TicTacToeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    num_iterations = 100  # Adjust as needed
    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        states, mcts_probs, rewards = self_play_game(net, num_simulations=1000)
        train_network(net, optimizer, states, mcts_probs, rewards, epochs=1)
