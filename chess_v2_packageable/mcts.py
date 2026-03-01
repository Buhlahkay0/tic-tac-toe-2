import math
import numpy as np
import torch
import random
import chess

from chess_game import ChessGame
from network import board_to_tensor, OUTPUT_DIM

# ── Deterministic move encoding ───────────────────────────────────────────────
# Follows AlphaZero's 4672-plane action space: from_square (64) × move_type (73)
#   move_type  0-55 : queen-type moves  (8 directions × 7 distances)
#   move_type 56-63 : knight moves      (8 offsets)
#   move_type 64-72 : underpromotions   (3 file-deltas × 3 pieces: N, B, R)
#
# Queen promotion moves fall into the queen-type branch (promotion=QUEEN is not
# an underpromotion). python-chess always sets a promotion piece for pawn moves
# that reach the back rank, so no ambiguity arises.

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


def move_to_index(move):
    """Deterministically map a chess.Move to an integer in [0, OUTPUT_DIM)."""
    from_sq = move.from_square
    to_sq   = move.to_square
    dr = (to_sq // 8) - (from_sq // 8)
    df = (to_sq %  8) - (from_sq %  8)

    if move.promotion and move.promotion != chess.QUEEN:
        # Underpromotion: pawn moves exactly one rank, file delta in {-1, 0, +1}
        dir_idx   = df + 1  # maps -1→0, 0→1, +1→2
        piece_idx = _UNDER_PROMO_MAP[move.promotion]
        move_type = 64 + dir_idx * 3 + piece_idx
    elif (abs(dr), abs(df)) in {(2, 1), (1, 2)}:
        # Knight move
        move_type = 56 + _KNIGHT_MOVE_MAP[(dr, df)]
    else:
        # Queen-type move (including queen promotions)
        unit = (
            0 if dr == 0 else (1 if dr > 0 else -1),
            0 if df == 0 else (1 if df > 0 else -1),
        )
        distance  = max(abs(dr), abs(df)) - 1
        move_type = _QUEEN_DIR_MAP[unit] * 7 + distance

    return from_sq * 73 + move_type


def index_to_move(index, chess_board):
    """
    Reconstruct a legal chess.Move from an action index and the current board.
    Falls back to a random legal move if the decoded move is not legal.
    """
    from_sq   = index // 73
    move_type = index  % 73
    from_rank = from_sq // 8
    from_file = from_sq  % 8

    if move_type < 56:
        direction = move_type // 7
        distance  = move_type %  7 + 1
        dr, df    = _QUEEN_DIRS[direction]
        to_rank   = from_rank + dr * distance
        to_file   = from_file + df * distance
        to_sq     = to_rank * 8 + to_file
        piece     = chess_board.piece_at(from_sq)
        promotion = None
        if piece and piece.piece_type == chess.PAWN and (to_rank == 7 or to_rank == 0):
            promotion = chess.QUEEN
        move = chess.Move(from_sq, to_sq, promotion=promotion)

    elif move_type < 64:
        dr, df  = _KNIGHT_MOVES[move_type - 56]
        to_sq   = (from_rank + dr) * 8 + (from_file + df)
        move    = chess.Move(from_sq, to_sq)

    else:
        under_idx = move_type - 64
        dir_idx   = under_idx // 3
        piece_idx = under_idx  % 3
        df        = dir_idx - 1  # 0→-1, 1→0, 2→+1
        piece     = chess_board.piece_at(from_sq)
        dr        = 1 if (piece and piece.color == chess.WHITE) else -1
        to_sq     = (from_rank + dr) * 8 + (from_file + df)
        move      = chess.Move(from_sq, to_sq, promotion=_UNDER_PROMO_PIECES[piece_idx])

    if move in chess_board.legal_moves:
        return move
    return random.choice(list(chess_board.legal_moves))


class MCTSNode:
    def __init__(self, game, parent=None, prior=0):
        self.game        = game
        self.parent      = parent
        self.children    = {}   # chess.Move -> MCTSNode
        self.visit_count = 0
        self.value_sum   = 0
        self.prior       = prior

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game.get_valid_moves())


class MCTS:
    def __init__(self, net, c_puct=1.0, num_simulations=100, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.net               = net
        self.c_puct            = c_puct
        self.num_simulations   = num_simulations
        self.dirichlet_alpha   = dirichlet_alpha    # controls noise shape (0.3 is standard for chess)
        self.dirichlet_epsilon = dirichlet_epsilon  # how much noise to mix in (0.25 is AlphaZero standard)

    def search(self, game, add_noise=False):
        root = MCTSNode(game)
        self.expand(root)
        if add_noise and root.children:
            self._add_dirichlet_noise(root)
        for _ in range(self.num_simulations):
            node        = root
            search_path = [node]
            while node.is_fully_expanded() and not node.game.is_terminal():
                move, node = self.select_child(node)
                search_path.append(node)
            if not node.game.is_terminal():
                self.expand(node)
                value = self.evaluate(node)
            else:
                winner = node.game.check_winner()
                if winner == 0:
                    value = 0
                else:
                    value = 1 if winner != self.get_current_player(node) else -1
            self.backpropagate(search_path, value)
        return root

    def _add_dirichlet_noise(self, root):
        moves   = list(root.children.keys())
        noise   = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        eps     = self.dirichlet_epsilon
        for move, eta in zip(moves, noise):
            child       = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * eta

    def select_child(self, node):
        best_score = -float('inf')
        best_move  = None
        best_child = None
        total_visits = sum(child.visit_count for child in node.children.values())
        for move, child in node.children.items():
            u     = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = child.q_value() + u
            if score > best_score:
                best_score = score
                best_move  = move
                best_child = child
        return best_move, best_child

    def expand(self, node):
        if node.game.is_terminal():
            return
        valid_moves  = node.game.get_valid_moves()
        board_tensor = board_to_tensor(node.game.board)
        with torch.no_grad():
            policy, _ = self.net(board_tensor)
        policy = policy.exp().cpu().numpy().flatten()

        move_priors = {}
        total_prior = 0
        for move in valid_moves:
            p = policy[move_to_index(move)]
            move_priors[move] = p
            total_prior += p
        if total_prior > 0:
            for move in move_priors:
                move_priors[move] /= total_prior
        else:
            uniform = 1.0 / len(valid_moves)
            for move in valid_moves:
                move_priors[move] = uniform

        for move in valid_moves:
            if move not in node.children:
                new_game = node.game.clone()
                new_game.make_move(move)
                node.children[move] = MCTSNode(new_game, parent=node, prior=move_priors[move])

    def evaluate(self, node):
        board_tensor = board_to_tensor(node.game.board)
        with torch.no_grad():
            _, value = self.net(board_tensor)
        return value.item()

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum   += value
            value             = -value

    def get_current_player(self, node):
        return 1 if node.game.board.turn == chess.WHITE else -1
