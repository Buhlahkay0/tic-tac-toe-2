import torch
import chess
from chess_game import ChessGame
from mcts import MCTS, move_to_index
from network import ChessNet, board_to_tensor, OUTPUT_DIM, device

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
        
        states.append(game.board.fen())
        players.append(1 if game.board.turn == chess.WHITE else -1)
        policy_vector = [0] * OUTPUT_DIM
        for move, prob in probs.items():
            idx = move_to_index(move)
            policy_vector[idx] = prob
        mcts_probs.append(policy_vector)
        
        best_move = max(visit_counts, key=visit_counts.get)
        game.make_move(best_move)
    
    winner = game.check_winner()
    rewards = []
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
            value_target = torch.tensor([[reward]], dtype=torch.float32).to(device)
            value_loss = (value - value_target).pow(2).mean()
            target_pi = torch.tensor(pi, dtype=torch.float32).unsqueeze(0).to(device)
            policy_loss = -torch.sum(target_pi * log_pi)
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(states):.4f}")

if __name__ == "__main__":
    net = ChessNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    try:
        num_iterations = int(input("Enter number of iterations (default 50): ") or "50")
    except ValueError:
        print("Invalid input. Defaulting to 50 iterations.")
        num_iterations = 50

    try:
        checkpoint_freq = int(input("Enter checkpoint save frequency (default 10): ") or "10")
    except ValueError:
        print("Invalid input. Defaulting to 10 iterations for checkpoint saving.")
        checkpoint_freq = 10

    checkpoint_path = "chess_model_checkpoint.pth"
    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        states, mcts_probs, rewards, players = self_play_game(net, num_simulations=100)
        train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1)
        if (iteration + 1) % checkpoint_freq == 0 or (iteration+1) == num_iterations:
            # Save both the model weights and the optimizer state.
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration+1}")
