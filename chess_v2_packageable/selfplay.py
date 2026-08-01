"""
Single-process self-play with cross-game GPU batching.

Many games run concurrently. Each engine tick advances every game's MCTS by
one simulation: all games descend to a leaf, the leaves that need a network
evaluation are stacked into ONE batched forward pass, and the results are
scattered back. With N parallel games the GPU sees batches of ~N positions
instead of single boards, which is what actually keeps it busy.

Training examples are (fen, policy_indices, policy_probs, value) tuples where
the policy is the root visit-count distribution (sparse, over legal moves)
and the value is the final game result from the perspective of the side to
move at that position.
"""

import time
from collections import Counter
from dataclasses import dataclass

import chess
import numpy as np
import torch

from encoding import move_to_index
from mcts import MCTS, Node

_MATERIAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}


def material_balance(board):
    """Material difference in pawns, positive = White ahead."""
    score = 0
    for piece_type, value in _MATERIAL.items():
        score += value * (len(board.pieces(piece_type, chess.WHITE))
                          - len(board.pieces(piece_type, chess.BLACK)))
    return score


def value_from_winner(winner, turn):
    """Game result (1/-1/0, White's perspective) → value for the side to move."""
    if winner == 0:
        return 0.0
    return 1.0 if (winner == 1) == (turn == chess.WHITE) else -1.0


@dataclass
class SelfPlayConfig:
    num_simulations: int = 200
    parallel_games: int = 64
    temperature_plies: int = 20   # sample moves ∝ visits for this many plies, then argmax
    max_game_plies: int = 200
    adjudication_margin: int = 2  # material lead (pawns) that wins a truncated game
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class _Slot:
    """One in-flight game."""

    def __init__(self):
        self.board = chess.Board()
        self.root = Node()
        self.needs_noise = True
        self.sims_done = 0
        self.ply = 0
        self.history = []  # (fen, policy_indices, policy_probs, turn) per move


class SelfPlayEngine:
    def __init__(self, net, device, config=None):
        self.net = net
        self.device = device
        self.cfg = config or SelfPlayConfig()
        self.mcts = MCTS(
            c_puct=self.cfg.c_puct,
            dirichlet_alpha=self.cfg.dirichlet_alpha,
            dirichlet_epsilon=self.cfg.dirichlet_epsilon,
        )

    def play(self, num_games, verbose=True):
        """
        Play `num_games` complete games and return (examples, stats).
        """
        cfg = self.cfg
        self.net.eval()
        started = min(cfg.parallel_games, num_games)
        slots = [_Slot() for _ in range(started)]
        examples = []
        stats = Counter()
        t0 = time.time()

        while slots:
            self._tick(slots)

            finished_indices = []
            for i, slot in enumerate(slots):
                if slot.sims_done < cfg.num_simulations:
                    continue
                if self._advance(slot, examples, stats):
                    finished_indices.append(i)

            for i in reversed(finished_indices):
                if started < num_games:
                    slots[i] = _Slot()
                    started += 1
                else:
                    slots.pop(i)

            if verbose and finished_indices and stats["games"] % 16 == 0:
                elapsed = time.time() - t0
                rate = stats["games"] / (elapsed / 60)
                print(f"  self-play: {stats['games']}/{num_games} games, "
                      f"{len(examples)} positions, {rate:.1f} games/min")

        stats["seconds"] = time.time() - t0
        return examples, stats

    def _tick(self, slots):
        """Advance every game by one MCTS simulation, batching network calls."""
        pending = []
        for slot in slots:
            path, result = self.mcts.run_to_leaf(slot.root, slot.board)
            if result[0] == "value":
                self.mcts.backpropagate(path, result[1])
                slot.sims_done += 1
            else:
                pending.append((slot, path, result[1], result[2], result[3]))

        if not pending:
            return

        planes = np.stack([p[2] for p in pending])
        batch = torch.from_numpy(planes).to(self.device)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda"):
                    log_policy, values = self.net(batch)
            else:
                log_policy, values = self.net(batch)
        policies = log_policy.float().exp().cpu().numpy()
        values = values.float().cpu().numpy().reshape(-1)

        for (slot, path, _, legal_moves, turn), policy, value in zip(pending, policies, values):
            leaf = path[-1]
            self.mcts.expand(leaf, legal_moves, turn, policy)
            if leaf is slot.root and slot.needs_noise:
                self.mcts.add_dirichlet_noise(slot.root)
                slot.needs_noise = False
            self.mcts.backpropagate(path, float(value))
            slot.sims_done += 1

    def _advance(self, slot, examples, stats):
        """Play the searched move. Returns True if the game ended."""
        cfg = self.cfg
        board = slot.board
        root = slot.root
        turn = board.turn

        visits = {m: c.visit_count for m, c in root.children.items()}
        total = sum(visits.values())
        moves = list(visits.keys())
        probs = np.array([visits[m] / total for m in moves])

        slot.history.append((
            board.fen(),
            [move_to_index(m, turn) for m in moves],
            probs.tolist(),
            turn,
        ))

        if slot.ply < cfg.temperature_plies:
            move = moves[np.random.choice(len(moves), p=probs)]
        else:
            move = max(visits, key=visits.get)

        board.push(move)
        slot.ply += 1

        # Reuse the chosen subtree as the next search root.
        slot.root = root.children[move]
        slot.sims_done = 0
        if slot.root.expanded:
            self.mcts.add_dirichlet_noise(slot.root)
            slot.needs_noise = False
        else:
            slot.needs_noise = True

        outcome = self._game_result(board, slot.ply)
        if outcome is None:
            return False

        winner, reason = outcome
        stats["games"] += 1
        stats["plies"] += slot.ply
        stats["white" if winner == 1 else "black" if winner == -1 else "draw"] += 1
        stats[f"by_{reason}"] += 1

        for fen, indices, move_probs, pos_turn in slot.history:
            examples.append((fen, indices, move_probs, value_from_winner(winner, pos_turn)))
        return True

    def _game_result(self, board, ply):
        """(winner, reason) if the game is over, else None. Winner: 1/-1/0."""
        if board.is_checkmate():
            return (1 if board.turn == chess.BLACK else -1), "checkmate"
        if board.is_stalemate() or board.is_insufficient_material():
            return 0, "draw"
        if board.halfmove_clock >= 100:
            return 0, "fifty_move"
        if board.is_repetition(3):
            return 0, "repetition"
        if ply >= self.cfg.max_game_plies:
            material = material_balance(board)
            if material >= self.cfg.adjudication_margin:
                return 1, "adjudication"
            if material <= -self.cfg.adjudication_margin:
                return -1, "adjudication"
            return 0, "adjudication"
        return None
