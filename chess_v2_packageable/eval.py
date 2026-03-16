"""
eval.py — Quick benchmark for the chess model.

Runs two tests:
  1. Model vs random mover  (no extra tools needed)
  2. Model vs Stockfish     (requires stockfish.exe in PATH or same folder)

A random mover is ~200 ELO.
Stockfish depth 1  is ~600-800 ELO.
Stockfish depth 2  is ~1000-1200 ELO.
Stockfish depth 3  is ~1400-1600 ELO.

If your model beats random consistently → it has learned something real.
If it beats Stockfish depth 1        → you're in 800 ELO territory.
"""

import os
import sys
import random
import chess
import chess.engine
import torch
from chess_game import ChessGame
from mcts import MCTS
from network import ChessNet, device
from play_v3 import print_board_with_coords


def load_model(checkpoint_path, num_simulations=200):
    net = ChessNet().to(device)
    if os.path.exists(checkpoint_path):
        data = torch.load(checkpoint_path, map_location=device)
        state = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
        net.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found — evaluating untrained model.")
    net.eval()
    return net


def model_move(mcts, game):
    root = mcts.search(game)
    visit_counts = {
        m: root.children[m].visit_count if m in root.children else 0
        for m in game.get_valid_moves()
    }
    return max(visit_counts, key=visit_counts.get)


def random_move(game, prevent_draw=True):
    moves = game.get_valid_moves()
    if prevent_draw:
        non_draw = [m for m in moves if not _would_draw(game, m)]
        return random.choice(non_draw if non_draw else moves)
    return random.choice(moves)


def _would_draw(game, move):
    test = game.clone()
    test.make_move(move)
    return test.board.can_claim_draw()


def play_one_game(net, num_simulations, model_is_white, prevent_draw=True):
    """
    Returns (result, final_fen) where result is 1 (win), -1 (loss), or 0 (draw).
    """
    game = ChessGame()
    mcts = MCTS(net, num_simulations=num_simulations)
    move_count = 0
    max_moves = 150

    while not game.is_terminal() and move_count < max_moves:
        if prevent_draw and game.board.can_claim_draw():
            break
        model_turn = (game.board.turn == chess.WHITE) == model_is_white
        if model_turn:
            move = model_move(mcts, game)
        else:
            move = random_move(game, prevent_draw=prevent_draw)
        game.make_move(move)
        move_count += 1

    winner = game.check_winner()
    model_color = 1 if model_is_white else -1
    if winner == model_color:
        return 1, game.board.fen()
    elif winner == -model_color:
        return -1, game.board.fen()
    return 0, game.board.fen()


def vs_random(net, num_simulations, num_games=20, prevent_draw=True):
    print(f"\n--- Model vs Random Mover ({num_games} games, {num_simulations} sims) ---")
    wins = losses = draws = 0
    final_fen = None
    for i in range(num_games):
        model_is_white = (i % 2 == 0)   # alternate colors
        result, fen = play_one_game(net, num_simulations, model_is_white, prevent_draw=prevent_draw)
        final_fen = fen
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        print(f"  Game {i+1:2d}: {'WIN' if result==1 else 'LOSS' if result==-1 else 'DRAW'}"
              f"  (model {'White' if model_is_white else 'Black'})")
        print_board_with_coords(chess.Board(fen), human_is_white=True)

    total = wins + losses + draws
    print(f"\nResult: {wins}W / {losses}L / {draws}D  "
          f"({100*wins/total:.0f}% wins)")
    if wins / total >= 0.80:
        print("  ✓ Strong vs random — model has learned basic chess.")
    elif wins / total >= 0.50:
        print("  ~ Inconsistent vs random — still early in training.")
    else:
        print("  ✗ Losing to random — model needs more training.")


def play_vs_stockfish(net, num_simulations, engine_path, depth, num_games=10, prevent_draw=True):
    print(f"\n--- Model vs Stockfish depth {depth} ({num_games} games, {num_simulations} sims) ---")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except FileNotFoundError:
        print(f"  Stockfish not found at '{engine_path}'. Skipping.")
        return

    wins = losses = draws = 0
    for i in range(num_games):
        model_is_white = (i % 2 == 0)
        game = ChessGame()
        mcts = MCTS(net, num_simulations=num_simulations)
        move_count = 0
        max_moves = 150

        while not game.is_terminal() and move_count < max_moves:
            if prevent_draw and game.board.can_claim_draw():
                break
            model_turn = (game.board.turn == chess.WHITE) == model_is_white
            if model_turn:
                move = model_move(mcts, game)
            else:
                sf_result = engine.play(game.board, chess.engine.Limit(depth=depth))
                move = sf_result.move
            game.make_move(move)
            move_count += 1

        winner = game.check_winner()
        model_color = 1 if model_is_white else -1
        if winner == model_color:
            outcome, wins = "WIN", wins + 1
        elif winner == -model_color:
            outcome, losses = "LOSS", losses + 1
        else:
            outcome, draws = "DRAW", draws + 1

        print(f"  Game {i+1:2d}: {outcome}"
              f"  (model {'White' if model_is_white else 'Black'})")
        print_board_with_coords(game.board, human_is_white=True)

    engine.quit()
    total = wins + losses + draws
    print(f"\nResult vs Stockfish depth {depth}: {wins}W / {losses}L / {draws}D  "
          f"({100*wins/total:.0f}% wins)")
    if wins / total >= 0.50:
        print(f"  ✓ Beating Stockfish depth {depth} — solid progress.")
    else:
        print(f"  ✗ Not beating Stockfish depth {depth} yet.")


def main():
    checkpoint = "chess_model_checkpoint.pth"
    try:
        num_simulations = int(input("Simulations per move for eval (default 200): ") or "200")
    except ValueError:
        num_simulations = 200

    try:
        num_games = int(input("Number of games vs random (default 20): ") or "20")
    except ValueError:
        num_games = 20

    prevent_draw = input("Prevent draw by repetition? (y/n, default y): ").strip().lower() != "n"

    net = load_model(checkpoint, num_simulations)

    # Always run random baseline
    vs_random(net, num_simulations, num_games=num_games, prevent_draw=prevent_draw)

    # Stockfish eval — try common locations
    sf_candidates = [
        "stockfish",                    # in PATH
        "stockfish.exe",
        os.path.join(os.path.dirname(__file__), "stockfish.exe"),
        os.path.join(os.path.dirname(__file__), "stockfish"),
    ]
    sf_path = next((p for p in sf_candidates if os.path.exists(p) or p in ("stockfish", "stockfish.exe")), None)

    run_sf = input("\nRun Stockfish eval? (y/n, requires stockfish in PATH or same folder): ").strip().lower()
    if run_sf == "y":
        try:
            depth = int(input("Stockfish depth (1=~700 ELO, 2=~1100 ELO, 3=~1500 ELO, default 1): ") or "1")
        except ValueError:
            depth = 1
        try:
            sf_games = int(input("Number of games vs Stockfish (default 10): ") or "10")
        except ValueError:
            sf_games = 10
        play_vs_stockfish(net, num_simulations, sf_path or "stockfish", depth, num_games=sf_games, prevent_draw=prevent_draw)


if __name__ == "__main__":
    main()
