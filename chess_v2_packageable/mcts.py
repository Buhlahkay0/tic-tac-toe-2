"""
PUCT Monte-Carlo tree search.

Sign convention (uniform everywhere): a node's value_sum accumulates values
from the perspective of the side to move AT that node. The network is trained
on the same convention (it always evaluates for the side to move), so its
output backpropagates directly. Selection therefore negates the child's Q,
because a child that is good for the opponent is bad for the parent.

Nodes hold no board state. During a simulation, moves are pushed onto the
game's real board on the way down and popped on the way back up, so no boards
are ever cloned and repetition/50-move detection sees the true game history.
"""

import math

import chess
import numpy as np
import torch

from encoding import board_to_planes, move_to_index


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = None  # None = not yet expanded; dict move -> Node after

    @property
    def expanded(self):
        return self.children is not None

    def q(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def terminal_value(board):
    """
    None if the position is not terminal, otherwise the game value from the
    perspective of the side to move (-1 mated, 0 drawn).
    """
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 100:  # 50-move rule
        return 0.0
    if board.is_repetition(3):      # threefold repetition
        return 0.0
    return None


class MCTS:
    def __init__(self, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def run_to_leaf(self, root, board):
        """
        Descend from root to a leaf, pushing moves onto `board` along the way
        and popping them before returning.

        Returns (path, result) where result is either
          ("value", v)                       — terminal leaf, v from the leaf
                                               mover's perspective, or
          ("eval", planes, legal_moves, turn) — leaf needs a network evaluation.
        """
        node = root
        path = [node]
        pushes = 0
        while node.expanded:
            move, node = self.select_child(node)
            board.push(move)
            pushes += 1
            path.append(node)

        tv = terminal_value(board)
        if tv is not None:
            result = ("value", tv)
        else:
            result = ("eval", board_to_planes(board), list(board.legal_moves), board.turn)

        for _ in range(pushes):
            board.pop()
        return path, result

    def select_child(self, node):
        sqrt_n = math.sqrt(max(1, node.visit_count))
        best_score = -float("inf")
        best_move = None
        best_child = None
        for move, child in node.children.items():
            q = -child.q()  # child's value is from the opponent's perspective
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def expand(self, node, legal_moves, turn, policy):
        """Attach children for all legal moves with priors from `policy` (4672 probs)."""
        priors = np.array([policy[move_to_index(m, turn)] for m in legal_moves],
                          dtype=np.float64)
        total = priors.sum()
        if total > 1e-8:
            priors /= total
        else:
            priors[:] = 1.0 / len(legal_moves)
        node.children = {m: Node(float(p)) for m, p in zip(legal_moves, priors)}

    def add_dirichlet_noise(self, root):
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_epsilon
        for move, eta in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * eta

    def backpropagate(self, path, value):
        """`value` is from the perspective of the side to move at the leaf."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def search(self, board, net, num_simulations, device, add_noise=False):
        """
        Blocking search for play/eval (one network call per simulation).
        Self-play uses SelfPlayEngine instead, which batches across games.
        Returns the root node; pick the child with the most visits.
        """
        root = Node()
        for _ in range(num_simulations):
            path, result = self.run_to_leaf(root, board)
            if result[0] == "value":
                value = result[1]
            else:
                _, planes, legal_moves, turn = result
                x = torch.from_numpy(planes).unsqueeze(0).to(device)
                with torch.no_grad():
                    log_policy, v = net(x)
                policy = log_policy[0].float().exp().cpu().numpy()
                leaf = path[-1]
                self.expand(leaf, legal_moves, turn, policy)
                if leaf is root and add_noise:
                    self.add_dirichlet_noise(root)
                value = float(v.item())
            self.backpropagate(path, value)
        return root
