"""
Benchmark the model against a random mover and Elo-limited Stockfish.

Stockfish's UCI_Elo mode is a calibrated strength limit (minimum ~1320),
which makes it a much better yardstick than depth caps:

    python eval.py --checkpoint chess_checkpoint.pth
    python eval.py --stockfish-elo 1320 1600 1900

Needs a `stockfish` binary on PATH (or pass --stockfish-path).
"""

import argparse
import random
import shutil

import chess
import chess.engine
import torch

from mcts import MCTS
from network import load_checkpoint, ChessNet, device
from selfplay import material_balance

MAX_PLIES = 300


def model_move(net, mcts, board, num_simulations):
    root = mcts.search(board, net, num_simulations, device)
    return max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]


def play_game(net, num_simulations, model_is_white, opponent_move):
    """Returns 1/-1/0 from the model's perspective."""
    board = chess.Board()
    mcts = MCTS()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        if (board.turn == chess.WHITE) == model_is_white:
            move = model_move(net, mcts, board, num_simulations)
        else:
            move = opponent_move(board)
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None and outcome.winner is not None:
        return 1 if (outcome.winner == chess.WHITE) == model_is_white else -1
    # Unfinished (ply cap): adjudicate by material for reporting.
    material = material_balance(board)
    if outcome is None and abs(material) >= 2:
        return 1 if (material > 0) == model_is_white else -1
    return 0


def run_match(net, num_simulations, num_games, opponent_move, label):
    wins = draws = losses = 0
    for i in range(num_games):
        result = play_game(net, num_simulations, model_is_white=(i % 2 == 0),
                           opponent_move=opponent_move)
        wins += result == 1
        draws += result == 0
        losses += result == -1
        print(f"  [{label}] game {i+1}/{num_games}: "
              f"W {wins} / D {draws} / L {losses}", end="\r")
    score = (wins + 0.5 * draws) / num_games
    print(f"\n  {label}: +{wins} ={draws} -{losses}  (score {score:.0%})")
    return score


def main():
    p = argparse.ArgumentParser(description="Evaluate the chess model")
    p.add_argument("--checkpoint", default="chess_checkpoint.pth")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--skip-random", action="store_true")
    p.add_argument("--stockfish-elo", type=int, nargs="*", default=[1320],
                   help="UCI_Elo levels to test against (Stockfish minimum is ~1320)")
    p.add_argument("--stockfish-path", default=None)
    args = p.parse_args()

    try:
        net, _ = load_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print("No checkpoint found — evaluating an untrained model.")
        net = ChessNet().to(device)
    net.eval()

    if not args.skip_random:
        print(f"\nModel vs random mover ({args.games} games):")
        run_match(net, args.simulations, args.games,
                  lambda board: random.choice(list(board.legal_moves)), "random")

    stockfish_path = args.stockfish_path or shutil.which("stockfish")
    if stockfish_path is None:
        print("\nStockfish not found on PATH — skipping engine matches.")
        return

    for elo in args.stockfish_elo:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        try:
            print(f"\nModel vs Stockfish @ {elo} Elo ({args.games} games):")
            run_match(
                net, args.simulations, args.games,
                lambda board: engine.play(board, chess.engine.Limit(time=0.1)).move,
                f"SF {elo}",
            )
        finally:
            engine.quit()


if __name__ == "__main__":
    main()
