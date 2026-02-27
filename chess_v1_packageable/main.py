import os
import torch
from network import ChessNet, device
from train import self_play_game, train_network

def main():
    # Create the network and optimizer, and move the network to GPU if available.
    net = ChessNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    checkpoint = "chess_model_checkpoint.pth"
    if os.path.exists(checkpoint):
        checkpoint_data = torch.load(checkpoint, map_location=device)
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            net.load_state_dict(checkpoint_data["model_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            print("Loaded checkpoint with model and optimizer state.")
        else:
            net.load_state_dict(checkpoint_data)
            print("Loaded old checkpoint with model weights only.")
    else:
        print("No checkpoint found. Starting from scratch.")
    
    # Prompt the user for number of iterations.
    try:
        num_iterations = int(input("Enter the number of iterations (default 200): ") or "200")
    except ValueError:
        print("Invalid input. Defaulting to 200 iterations.")
        num_iterations = 200

    # Prompt the user for number of simulations.
    try:
        num_simulations = int(input("Enter the number of simulations for self-play (default 200): ") or "200")
    except ValueError:
        print("Invalid input. Defaulting to 200 simulations.")
        num_simulations = 200

    # Prompt the user for checkpoint frequency.
    try:
        checkpoint_freq = int(input("Enter checkpoint save frequency (default 10): ") or "10")
    except ValueError:
        print("Invalid input. Defaulting to checkpoint saving every 10 iterations.")
        checkpoint_freq = 10

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")
        # Run one self-play game to generate training examples.
        states, mcts_probs, rewards, players = self_play_game(net, num_simulations=num_simulations)
        # Train the network on the new examples.
        train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1)
        # Save the updated model and optimizer state based on the specified frequency.
        if (iteration + 1) % checkpoint_freq == 0 or (iteration+1) == num_iterations:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint)
            print(f"Checkpoint saved at iteration {iteration+1}")

if __name__ == "__main__":
    main()
