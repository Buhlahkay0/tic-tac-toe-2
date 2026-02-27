import os
import torch
from network import ChessNet
from train import self_play_game, train_network

def main():
    # Create the network and optimizer.
    net = ChessNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    checkpoint = "chess_model_checkpoint.pth"
    if os.path.exists(checkpoint):
        net.load_state_dict(torch.load(checkpoint))
        print("Checkpoint loaded from", checkpoint)
    else:
        print("No checkpoint found. Starting from scratch.")
    
    num_iterations = 200  # Adjust as needed.
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")
        # Run one self-play game to generate training examples.
        states, mcts_probs, rewards, players = self_play_game(net, num_simulations=200)
        # Train the network on the new examples.
        train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1)
        # Save the updated model.
        torch.save(net.state_dict(), checkpoint)
        print(f"Checkpoint saved at iteration {iteration+1}")

if __name__ == "__main__":
    main()
