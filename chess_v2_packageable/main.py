import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import chess
import torch
import torch.multiprocessing as tmp
from network import ChessNet, device
from play_v3 import print_board_with_coords
from train import _run_self_play, train_network

REPLAY_BUFFER_SIZE      = 10_000
MIN_BUFFER_FOR_TRAINING = 512
BATCH_SIZE              = 512


def main():
    net       = ChessNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

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

    try:
        eval_batch_size = int(input("Enter MCTS eval batch size (default 16): ") or "16")
    except ValueError:
        print("Invalid input. Defaulting to 16.")
        eval_batch_size = 16

    cpu_count = os.cpu_count() or 4
    default_workers = max(1, cpu_count - 1)
    try:
        num_workers = int(input(f"Enter number of parallel self-play workers (default {default_workers}, max {cpu_count - 1}): ") or str(default_workers))
        num_workers = max(1, min(num_workers, cpu_count - 1))
    except ValueError:
        print(f"Invalid input. Defaulting to {default_workers} workers.")
        num_workers = default_workers

    print(f"Running {num_workers} self-play workers on CPU, training on {device}.")

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    white_wins = black_wins = draw_count = 0

    # spawn context required on Windows for CUDA safety in subprocesses
    mp_ctx = tmp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as pool:
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration+1}/{num_iterations}")

            cpu_weights = {k: v.cpu() for k, v in net.state_dict().items()}
            worker_args = [(cpu_weights, num_simulations, max_moves, eval_batch_size)] * num_workers

            # All workers run in parallel; we block until all finish
            game_results = list(pool.map(_run_self_play, worker_args))

            final_fen = None
            for states, mcts_probs, rewards, players, winner, game_fen in game_results:
                if winner == 1:
                    white_wins += 1
                elif winner == -1:
                    black_wins += 1
                else:
                    draw_count += 1
                replay_buffer.extend(zip(states, mcts_probs, rewards, players))
                final_fen = game_fen

            print(f"  Stats — White: {white_wins}, Black: {black_wins}, Draws: {draw_count}")

            if final_fen:
                print_board_with_coords(chess.Board(final_fen), human_is_white=True)

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
