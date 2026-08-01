"""
Board and move encoding for the chess network.

Everything is encoded from the perspective of the side to move: when Black is
to move, the board is mirrored vertically and colors are swapped, so the
network always sees "my pieces moving up the board". The policy and value
outputs are therefore always from the mover's point of view, and each pattern
only has to be learned once instead of once per color.

Input planes (17 x 8 x 8):
   0-5  : my pawns, knights, bishops, rooks, queens, king
   6-11 : their pawns, knights, bishops, rooks, queens, king
  12-15 : castling rights (mine K-side, mine Q-side, theirs K-side, theirs Q-side)
  16    : en passant target square

Action space (4672 = 64 from-squares x 73 move types), AlphaZero-style:
  move type  0-55 : queen-type moves (8 directions x 7 distances)
  move type 56-63 : knight moves (8 offsets)
  move type 64-72 : underpromotions (3 file-deltas x 3 pieces: N, B, R)

Queen promotions fall into the queen-type branch. Move squares are mirrored
the same way as the board when Black is to move, so a move index always means
the same thing relative to the mover.
"""

import chess
import numpy as np

PLANES = 17
OUTPUT_DIM = 4672

_QUEEN_DIRS = [
    ( 1,  0),  # N
    ( 1,  1),  # NE
    ( 0,  1),  # E
    (-1,  1),  # SE
    (-1,  0),  # S
    (-1, -1),  # SW
    ( 0, -1),  # W
    ( 1, -1),  # NW
]
_QUEEN_DIR_MAP = {d: i for i, d in enumerate(_QUEEN_DIRS)}

_KNIGHT_MOVES = [
    ( 2,  1), ( 1,  2), (-1,  2), (-2,  1),
    (-2, -1), (-1, -2), ( 1, -2), ( 2, -1),
]
_KNIGHT_MOVE_MAP = {m: i for i, m in enumerate(_KNIGHT_MOVES)}

_UNDER_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDER_PROMO_MAP = {p: i for i, p in enumerate(_UNDER_PROMO_PIECES)}


def board_to_planes(board):
    """Encode a board as a (17, 8, 8) float32 array from the mover's perspective."""
    arr = np.zeros((PLANES, 8, 8), dtype=np.float32)
    us = board.turn
    flip = us == chess.BLACK

    for square, piece in board.piece_map().items():
        if flip:
            square = chess.square_mirror(square)
        ch = (piece.piece_type - 1) + (0 if piece.color == us else 6)
        arr[ch, square // 8, square % 8] = 1.0

    if board.has_kingside_castling_rights(us):
        arr[12].fill(1.0)
    if board.has_queenside_castling_rights(us):
        arr[13].fill(1.0)
    if board.has_kingside_castling_rights(not us):
        arr[14].fill(1.0)
    if board.has_queenside_castling_rights(not us):
        arr[15].fill(1.0)

    if board.ep_square is not None:
        ep = chess.square_mirror(board.ep_square) if flip else board.ep_square
        arr[16, ep // 8, ep % 8] = 1.0

    return arr


def move_to_index(move, turn):
    """Map a chess.Move to an action index, from the perspective of `turn`."""
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq   = chess.square_mirror(move.to_square)
    else:
        from_sq = move.from_square
        to_sq   = move.to_square

    dr = (to_sq // 8) - (from_sq // 8)
    df = (to_sq %  8) - (from_sq %  8)

    if move.promotion and move.promotion != chess.QUEEN:
        # Underpromotion: after mirroring the pawn always advances one rank.
        dir_idx   = df + 1  # -1→0, 0→1, +1→2
        piece_idx = _UNDER_PROMO_MAP[move.promotion]
        move_type = 64 + dir_idx * 3 + piece_idx
    elif (abs(dr), abs(df)) in {(2, 1), (1, 2)}:
        move_type = 56 + _KNIGHT_MOVE_MAP[(dr, df)]
    else:
        unit = (
            0 if dr == 0 else (1 if dr > 0 else -1),
            0 if df == 0 else (1 if df > 0 else -1),
        )
        distance  = max(abs(dr), abs(df)) - 1
        move_type = _QUEEN_DIR_MAP[unit] * 7 + distance

    return from_sq * 73 + move_type
