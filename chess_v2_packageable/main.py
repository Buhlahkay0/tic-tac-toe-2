import os
import random
from collections import deque

import torch
from network import ChessNet, device
from train import self_play_game, train_network

REPLAY_BUFFER_SIZE     = 10_000
MIN_BUFFER_FOR_TRAINING = 512
BATCH_SIZE             = 512


def main():
    net       = ChessNet().to(device)
    net       = torch.compile(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scaler    = torch.cuda.GradScaler() if device.type == "cuda" else None

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

    try:
        num_iterations = int(input("Enter the number of iterations (default 200): ") or "200")
    except ValueError:
        print("Invalid input. Defaulting to 200 iterations.")
        num_iterations = 200

    try:
        num_simulations = int(input("Enter the number of simulations for self-play (default 200): ") or "200")
    except ValueError:
        print("Invalid input. Defaulting to 200 simulations.")
        num_simulations = 200

    try:
        checkpoint_freq = int(input("Enter checkpoint save frequency (default 10): ") or "10")
    except ValueError:
        print("Invalid input. Defaulting to checkpoint saving every 10 iterations.")
        checkpoint_freq = 10

    try:
        max_moves = int(input("Enter the maximum moves per game (default 150): ") or "150")
    except ValueError:
        print("Invalid input. Defaulting to 150 moves.")
        max_moves = 150

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    white_wins = black_wins = draw_count = 0

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        states, mcts_probs, rewards, players, winner = self_play_game(
            net, num_simulations=num_simulations, max_moves=max_moves
        )

        if winner == 1:
            white_wins += 1
        elif winner == -1:
            black_wins += 1
        else:
            draw_count += 1
        print(f"  Stats — White: {white_wins}, Black: {black_wins}, Draws: {draw_count}")

        # Add this game's data to the replay buffer
        replay_buffer.extend(zip(states, mcts_probs, rewards, players))

        # Train from a random batch once the buffer is large enough
        if len(replay_buffer) >= MIN_BUFFER_FOR_TRAINING:
            batch_size = min(BATCH_SIZE, len(replay_buffer))
            batch      = random.sample(replay_buffer, batch_size)
            b_states, b_probs, b_rewards, b_players = zip(*batch)
            train_network(
                net, optimizer,
                list(b_states), list(b_probs), list(b_rewards), list(b_players),
                epochs=1, scaler=scaler,
            )
        else:
            remaining = MIN_BUFFER_FOR_TRAINING - len(replay_buffer)
            print(
                f"  Buffer filling... {len(replay_buffer)}/{MIN_BUFFER_FOR_TRAINING} "
                f"({remaining} more needed before training starts)"
            )

        if (iteration + 1) % checkpoint_freq == 0 or (iteration + 1) == num_iterations:
            torch.save(
                {'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                checkpoint,
            )
            print(f"Checkpoint saved at iteration {iteration+1}")


if __name__ == "__main__":
    main()
