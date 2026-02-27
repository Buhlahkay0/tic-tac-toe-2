import math
import numpy as np
import torch
import random
import chess

from chess_game import ChessGame
from network import board_to_tensor, OUTPUT_DIM

def move_to_index(move):
    """
    A simple move-to-index mapping using a hash of the UCI string.
    In a production system, you would design a more structured encoding.
    """
    return hash(move.uci()) % OUTPUT_DIM

def index_to_move(index, chess_board):
    """
    Given an index and a chess board, return a legal move that maps to the index.
    This is a simple lookup over legal moves; if none match, returns a random move.
    """
    for move in chess_board.legal_moves:
        if move_to_index(move) == index:
            return move
    return random.choice(list(chess_board.legal_moves))

class MCTSNode:
    def __init__(self, game, parent=None, prior=0):
        self.game = game  # Instance of ChessGame
        self.parent = parent
        self.children = {}  # Mapping: chess.Move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def is_fully_expanded(self):
        # A node is fully expanded if all legal moves have been added as children.
        return len(self.children) == len(self.game.get_valid_moves())

class MCTS:
    def __init__(self, net, c_puct=1.0, num_simulations=100):
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, game):
        root = MCTSNode(game)
        self.expand(root)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            # Selection: traverse down until a leaf node.
            while node.is_fully_expanded() and not node.game.is_terminal():
                move, node = self.select_child(node)
                search_path.append(node)
            # If not terminal, expand the node.
            if not node.game.is_terminal():
                self.expand(node)
                value = self.evaluate(node)
            else:
                # Terminal state evaluation.
                winner = node.game.check_winner()
                if winner == 0:
                    value = 0
                else:
                    # Since the move that ended the game flipped the turn,
                    # the perspective of the terminal node is inverted.
                    value = 1 if winner != self.get_current_player(node) else -1
            self.backpropagate(search_path, value)
        return root

    def select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_child = None
        total_visits = sum(child.visit_count for child in node.children.values())
        for move, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = child.q_value() + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def expand(self, node):
        if node.game.is_terminal():
            return
        valid_moves = node.game.get_valid_moves()
        board_tensor = board_to_tensor(node.game.board)
        with torch.no_grad():
            policy, _ = self.net(board_tensor)
        policy = policy.exp().cpu().numpy().flatten()
        # Build a dictionary of priors for legal moves.
        move_priors = {}
        total_prior = 0
        for move in valid_moves:
            idx = move_to_index(move)
            p = policy[idx]
            move_priors[move] = p
            total_prior += p
        # Normalize the priors.
        if total_prior > 0:
            for move in move_priors:
                move_priors[move] /= total_prior
        else:
            uniform = 1 / len(valid_moves)
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
            node.value_sum += value
            value = -value

    def get_current_player(self, node):
        # Using python-chess, board.turn is True for White, False for Black.
        return 1 if node.game.board.turn == chess.WHITE else -1
