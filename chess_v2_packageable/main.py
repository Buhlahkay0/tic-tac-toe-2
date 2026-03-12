import os
import random
from collections import deque

import chess
import torch
import torch.multiprocessing as mp

from network import ChessNet, device
from play_v3 import print_board_with_coords
from train import train_network

REPLAY_BUFFER_SIZE      = 10_000
MIN_BUFFER_FOR_TRAINING = 512
BATCH_SIZE              = 512


def _worker(weights_dict, num_simulations, max_moves, result_queue):
    """
    Runs one self-play game on CPU and puts the result into result_queue.
    Must be a module-level function for Windows multiprocessing (spawn method).
    """
    from network import ChessNet
    from train import self_play_game
    net = ChessNet()           # CPU — each worker gets its own copy of the weights
    net.load_state_dict(weights_dict)
    net.eval()
    result = self_play_game(
        net, num_simulations=num_simulations, max_moves=max_moves,
        verbose=False, device=torch.device("cpu"),
    )
    result_queue.put(result)


def main():
    net       = ChessNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Load checkpoint BEFORE compiling — compiled models prefix keys with _orig_mod.
    # which would cause mismatches if loaded after torch.compile.
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

    net = torch.compile(net, backend="eager")

    try:
        num_iterations = int(input("Enter the number of iterations (default 200): ") or "200")
    except ValueError:
        print("Invalid input. Defaulting to 200 iterations.")
        num_iterations = 200

    try:
        num_simulations = int(input("Enter the number of simulations for self-play (default 100): ") or "100")
    except ValueError:
        print("Invalid input. Defaulting to 100 simulations.")
        num_simulations = 100

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
        num_workers = int(input("Enter number of parallel self-play workers (default 4): ") or "4")
    except ValueError:
        print("Invalid input. Defaulting to 4 workers.")
        num_workers = 4

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    white_wins = black_wins = draw_count = 0

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations} — running {num_workers} games in parallel")

        # Copy weights to CPU for workers. Strip the _orig_mod. prefix that
        # torch.compile adds so workers can load into an uncompiled ChessNet.
        cpu_weights = {
            k.replace("_orig_mod.", ""): v.cpu()
            for k, v in net.state_dict().items()
        }

        # Spawn worker processes
        result_queue = mp.Queue()
        processes    = []
        for _ in range(num_workers):
            p = mp.Process(
                target=_worker,
                args=(cpu_weights, num_simulations, max_moves, result_queue),
            )
            p.start()
            processes.append(p)

        # Collect results as workers finish
        all_results = [result_queue.get() for _ in range(num_workers)]
        for p in processes:
            p.join()

        # Process results from all games this iteration
        last_final_fen = None
        for i, (states, mcts_probs, rewards, players, winner, final_fen) in enumerate(all_results):
            result_str = {1: "White wins", -1: "Black wins", 0: "Draw", None: "Unfinished"}.get(winner)
            print(f"  Game {i+1}: {result_str} ({len(states)} moves)")

            if winner == 1:
                white_wins += 1
            elif winner == -1:
                black_wins += 1
            else:
                draw_count += 1

            replay_buffer.extend(zip(states, mcts_probs, rewards, players))
            last_final_fen = final_fen

        print(f"  Stats — White: {white_wins}, Black: {black_wins}, Draws: {draw_count}")

        # Print final board from the last game
        if last_final_fen:
            print_board_with_coords(chess.Board(last_final_fen), human_is_white=True)

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
