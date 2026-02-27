import torch
import chess
from chess_game import ChessGame
from mcts import MCTS, move_to_index
from network import ChessNet, board_to_tensor, OUTPUT_DIM

def self_play_game(net, num_simulations=100):
    """
    Runs a single self-play game using MCTS for move selection.
    Returns:
      - states: list of board positions stored as FEN strings.
      - mcts_probs: list of fixed-dimension (OUTPUT_DIM) probability vectors.
      - rewards: list of rewards from the perspective of the player making the move.
      - players: list of player indicators (1 for White, -1 for Black) corresponding to each state.
    """
    game = ChessGame()
    mcts = MCTS(net, num_simulations=num_simulations)
    states, mcts_probs, players = [], [], []
    
    while not game.is_terminal():
        root = mcts.search(game)
        valid_moves = game.get_valid_moves()
        visit_counts = {}
        for move in valid_moves:
            visit_counts[move] = root.children[move].visit_count if move in root.children else 0
        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            probs = {move: count / total_visits for move, count in visit_counts.items()}
        else:
            uniform_prob = 1 / len(valid_moves)
            probs = {move: uniform_prob for move in valid_moves}
        
        # Record the state (using FEN), the current player, and the policy vector.
        states.append(game.board.fen())
        players.append(1 if game.board.turn == chess.WHITE else -1)
        policy_vector = [0] * OUTPUT_DIM
        for move, prob in probs.items():
            idx = move_to_index(move)
            policy_vector[idx] = prob
        mcts_probs.append(policy_vector)
        
        # Select the move with the highest visit count.
        best_move = max(visit_counts, key=visit_counts.get)
        game.make_move(best_move)
    
    winner = game.check_winner()
    rewards = []
    # Assign rewards from the perspective of the player who made each move.
    for player in players:
        if winner == 0:
            rewards.append(0)
        else:
            rewards.append(1 if winner == player else -1)
    return states, mcts_probs, rewards, players

def train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1):
    """
    Trains the network on self-play data.
    Each training example consists of:
      - A board state (converted from FEN to tensor)
      - A target policy (mcts_probs)
      - A target value (reward)
    """
    net.train()
    for epoch in range(epochs):
        total_loss = 0
        for fen, pi, reward, player in zip(states, mcts_probs, rewards, players):
            board = chess.Board(fen)
            board_tensor = board_to_tensor(board)  # Shape: [1, 12, 8, 8]
            optimizer.zero_grad()
            log_pi, value = net(board_tensor)
            value_target = torch.tensor([[reward]], dtype=torch.float32)
            value_loss = (value - value_target).pow(2).mean()
            target_pi = torch.tensor(pi, dtype=torch.float32).unsqueeze(0)
            policy_loss = -torch.sum(target_pi * log_pi)
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(states):.4f}")

if __name__ == "__main__":
    net = ChessNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_iterations = 50  # Adjust the number of iterations as needed.
    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        states, mcts_probs, rewards, players = self_play_game(net, num_simulations=100)
        train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1)
        # Optionally, save a checkpoint after each iteration.
        torch.save(net.state_dict(), "chess_model_checkpoint.pth")
