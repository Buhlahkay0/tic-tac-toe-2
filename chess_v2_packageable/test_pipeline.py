"""
Correctness tests for the encoding, MCTS conventions, and self-play engine.

Run:  python test_pipeline.py
"""

import random

import chess
import numpy as np
import torch

from encoding import board_to_planes, move_to_index, PLANES, OUTPUT_DIM
from mcts import MCTS, terminal_value
from network import ChessNet
from selfplay import SelfPlayEngine, SelfPlayConfig, value_from_winner
from train import train_step

CPU = torch.device("cpu")


def random_board(seed, max_plies=80):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(rng.randrange(max_plies)):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    return board


def test_move_indices_unique_and_in_range():
    for seed in range(300):
        board = random_board(seed)
        indices = [move_to_index(m, board.turn) for m in board.legal_moves]
        assert all(0 <= i < OUTPUT_DIM for i in indices)
        assert len(set(indices)) == len(indices), f"index collision (seed {seed})"


def test_mirror_symmetry():
    """A position and its color-mirror must produce identical planes and move indices."""
    for seed in range(300):
        board = random_board(seed)
        mirrored = board.mirror()  # flips colors, ranks, turn, castling, ep
        assert np.array_equal(board_to_planes(board), board_to_planes(mirrored)), \
            f"plane mismatch (seed {seed})"
        for move in board.legal_moves:
            mirror_move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
            assert move_to_index(move, board.turn) == \
                   move_to_index(mirror_move, mirrored.turn), \
                f"move index mismatch for {move} (seed {seed})"


def test_planes_shape_and_start_position():
    planes = board_to_planes(chess.Board())
    assert planes.shape == (PLANES, 8, 8)
    assert planes[0].sum() == 8   # my pawns
    assert planes[6].sum() == 8   # their pawns
    assert planes[12:16].sum() == 4 * 64  # all castling rights
    assert planes[16].sum() == 0  # no en passant


def test_terminal_values():
    mated = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate
        mated.push(chess.Move.from_uci(uci))
    assert mated.is_checkmate()
    assert terminal_value(mated) == -1.0  # side to move is mated
    assert terminal_value(chess.Board()) is None


def _mate_in_one_net():
    torch.manual_seed(0)
    net = ChessNet(channels=16, blocks=1).to(CPU)
    net.eval()
    return net


def test_mcts_finds_mate_in_one_as_white():
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]:
        board.push(chess.Move.from_uci(uci))
    root = MCTS().search(board, _mate_in_one_net(), 150, CPU)
    best = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    assert best == chess.Move.from_uci("h5f7"), f"expected Qxf7#, got {best}"
    assert board.fen() == chess.Board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    ).fen(), "search must leave the board untouched"


def test_mcts_finds_mate_in_one_as_black():
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4"]:
        board.push(chess.Move.from_uci(uci))
    root = MCTS().search(board, _mate_in_one_net(), 150, CPU)
    best = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    assert best == chess.Move.from_uci("d8h4"), f"expected Qh4#, got {best}"


def test_value_from_winner_orientation():
    assert value_from_winner(1, chess.WHITE) == 1.0
    assert value_from_winner(1, chess.BLACK) == -1.0
    assert value_from_winner(-1, chess.WHITE) == -1.0
    assert value_from_winner(-1, chess.BLACK) == 1.0
    assert value_from_winner(0, chess.WHITE) == 0.0


def test_selfplay_engine_and_training_smoke():
    torch.manual_seed(0)
    np.random.seed(0)
    net = ChessNet(channels=16, blocks=1).to(CPU)
    config = SelfPlayConfig(num_simulations=16, parallel_games=2,
                            temperature_plies=4, max_game_plies=16)
    engine = SelfPlayEngine(net, CPU, config)
    examples, stats = engine.play(2, verbose=False)

    assert stats["games"] == 2
    assert len(examples) > 0
    for fen, indices, probs, value in examples:
        chess.Board(fen)  # must parse
        assert len(indices) == len(probs)
        assert abs(sum(probs) - 1.0) < 1e-6
        assert value in (-1.0, 0.0, 1.0)

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    policy_loss, value_loss = train_step(net, optimizer, examples, CPU)
    assert np.isfinite(policy_loss) and np.isfinite(value_loss)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
