import math
from network import board_to_tensor
import numpy as np
import torch

class MCTSNode:
    def __init__(self, game, parent=None, prior=0):
        self.game = game
        self.parent = parent
        self.children = {}  # move (tuple) -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def q_value(self):
        # Average value of this node.
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def is_fully_expanded(self):
        # A node is fully expanded if every valid move has a child.
        return len(self.children) == len(self.game.get_valid_moves())

class MCTS:
    def __init__(self, net, c_puct=1.0, num_simulations=50):
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, game):
        root = MCTSNode(game)
        # Expand the root using the network.
        self.expand(root)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse the tree until a leaf is reached.
            while node.is_fully_expanded() and not node.game.is_terminal():
                move, node = self.select_child(node)
                search_path.append(node)
            # If the node is not terminal, expand it.
            if not node.game.is_terminal():
                self.expand(node)
                # Evaluate the new node using the network.
                value = self.evaluate(node)
            else:
                # If terminal, use the game outcome.
                winner = node.game.check_winner()
                if winner == 0:  # draw
                    value = 0
                else:
                    # Value from the perspective of the node’s current player.
                    value = 1 if winner == node.game.current_player else -1

            # Backpropagate the evaluation up the tree.
            self.backpropagate(search_path, value)
        return root

    def select_child(self, node):
        # Use the PUCT formula to select the next child.
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
        # Convert the board state to tensor.
        board_tensor = board_to_tensor(node.game.board, node.game.current_player)
        with torch.no_grad():
            policy, _ = self.net(board_tensor)
        # Convert log probabilities back to probabilities.
        policy = policy.exp().cpu().numpy().flatten()
        # Mask invalid moves.
        mask = np.zeros(9, dtype=bool)
        for (i, j) in valid_moves:
            mask[i * 3 + j] = True
        policy = policy * mask
        sum_policy = np.sum(policy)
        if sum_policy > 0:
            policy /= sum_policy
        else:
            # If the network gives zero probability for all moves, use uniform probabilities.
            policy = mask.astype(np.float32) / np.sum(mask)
        # Create a child for each valid move.
        for (i, j) in valid_moves:
            move = (i, j)
            if move not in node.children:
                prior = policy[i * 3 + j]
                new_game = node.game.clone()
                new_game.make_move(move)
                node.children[move] = MCTSNode(new_game, parent=node, prior=prior)

    def evaluate(self, node):
        # Evaluate the node using the network.
        board_tensor = board_to_tensor(node.game.board, node.game.current_player)
        with torch.no_grad():
            _, value = self.net(board_tensor)
        return value.item()

    def backpropagate(self, search_path, value):
        # Propagate the evaluation value back up the search path.
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # alternate the perspective
