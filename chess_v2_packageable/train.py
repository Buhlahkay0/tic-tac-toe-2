import random
import torch
import torch.nn.functional as F
import chess
import numpy as np
from chess_game import ChessGame
from mcts import MCTS, move_to_index
from network import ChessNet, board_to_tensor, boards_to_batch, OUTPUT_DIM, device
from play_v3 import print_board_with_coords

TEMPERATURE_MOVES = 15  # use temperature sampling for the first N moves of each game


def _select_move_with_temperature(visit_counts):
    """Sample a move proportionally to visit counts (exploration)."""
    moves  = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    probs  = counts / counts.sum()
    return moves[np.random.choice(len(moves), p=probs)]


def self_play_game(net, num_simulations=100, max_moves=100, verbose=True, eval_batch_size=16, device=None):
    """
    Runs a single self-play game using MCTS for move selection.
    States and player labels are recorded BEFORE each move so they correspond
    to the position that generated the policy vector.
    Returns training data, the final winner, and the final board FEN.
    Set verbose=False to suppress all output.
    """
    game  = ChessGame()
    mcts  = MCTS(net, num_simulations=num_simulations, eval_batch_size=eval_batch_size, device=device)
    states, mcts_probs, players = [], [], []
    move_count = 0

    while not game.is_terminal():
        if move_count >= max_moves:
            if verbose:
                print("Game ended: move limit reached.")
            break

        root        = mcts.search(game, add_noise=True)
        valid_moves = game.get_valid_moves()
        visit_counts = {
            move: root.children[move].visit_count if move in root.children else 0
            for move in valid_moves
        }

        if not visit_counts:
            if verbose:
                print("No valid moves left. Game over.")
            break

        # Filter out moves that would immediately allow a draw claim, unless
        # every legal move does so (forced draw).
        non_draw_moves = {}
        for move, count in visit_counts.items():
            test_game = game.clone()
            test_game.make_move(move)
            if not test_game.board.can_claim_draw():
                non_draw_moves[move] = count
        if non_draw_moves:
            visit_counts = non_draw_moves
        else:
            if verbose:
                print("Game ended: draw unavoidable.")
            break

        # Temperature sampling for early moves, greedy afterwards.
        if move_count < TEMPERATURE_MOVES:
            best_move = _select_move_with_temperature(visit_counts)
        else:
            best_move = max(visit_counts, key=visit_counts.get)

        # Record state and active player BEFORE making the move so the stored
        # FEN matches the position that produced the visit-count policy vector.
        states.append(game.board.fen())
        players.append(1 if game.board.turn == chess.WHITE else -1)

        total_visits  = sum(visit_counts.values())
        policy_vector = [0.0] * OUTPUT_DIM
        for move, count in visit_counts.items():
            policy_vector[move_to_index(move)] = count / total_visits
        mcts_probs.append(policy_vector)

        game.make_move(best_move)
        move_count += 1

    winner    = game.check_winner()
    final_fen = game.board.fen()
    if verbose:
        result_str = {1: "White wins", -1: "Black wins", 0: "Draw", None: "Unfinished"}.get(winner)
        print(f"Game Over! Result: {result_str}")

    rewards = []
    for player in players:
        if winner is None or winner == 0:
            rewards.append(0)
        else:
            rewards.append(1 if winner == player else -1)

    return states, mcts_probs, rewards, players, winner, final_fen


def _run_self_play(args):
    """
    Top-level worker function for multiprocessing (must be at module level for
    Windows spawn to pickle it). Runs one self-play game entirely on CPU so that
    N workers can run in parallel without touching the GPU.
    """
    cpu_state_dict, num_simulations, max_moves, eval_batch_size = args
    cpu = torch.device('cpu')
    net = ChessNet()          # created on CPU — no CUDA needed in worker
    net.load_state_dict(cpu_state_dict)
    net.eval()
    return self_play_game(
        net, num_simulations=num_simulations, max_moves=max_moves,
        verbose=False, eval_batch_size=eval_batch_size, device=cpu,
    )


def train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1, scaler=None):
    """
    Trains the network on a batch of self-play data using a single forward pass
    over the entire batch rather than one pass per sample.
    """
    net.train()

    # Build all inputs on CPU then move in one transfer
    board_batch = boards_to_batch([chess.Board(fen) for fen in states]).to(device)
    target_pi   = torch.tensor(mcts_probs, dtype=torch.float32, device=device)
    target_v    = torch.tensor(rewards,    dtype=torch.float32, device=device).unsqueeze(1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            log_pi, value = net(board_batch)
            value_loss  = F.mse_loss(value, target_v)
            policy_loss = -(target_pi * log_pi).sum(dim=1).mean()
            loss        = value_loss + policy_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    net       = ChessNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    checkpoint_path = "chess_model_checkpoint.pth"

    num_iterations = int(input("Enter number of iterations (default 50): ") or "50")
    checkpoint_freq = int(input("Enter checkpoint save frequency (default 10): ") or "10")

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        states, mcts_probs, rewards, players, _, _fen = self_play_game(net, num_simulations=100)
        train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1)
        if (iteration + 1) % checkpoint_freq == 0 or (iteration + 1) == num_iterations:
            torch.save(
                {'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                checkpoint_path,
            )
            print(f"Checkpoint saved at iteration {iteration+1}")
