import torch
import chess
from chess_game import ChessGame
from mcts import MCTS, move_to_index
from network import ChessNet, board_to_tensor, OUTPUT_DIM, device


def self_play_game(net, num_simulations=100, max_moves=140):
    """
    Runs a single self-play game using MCTS for move selection.
    States and player labels are recorded BEFORE each move so they correspond
    to the position that generated the policy vector.
    Returns training data plus the final winner (1, -1, 0, or None).
    """
    game  = ChessGame()
    mcts  = MCTS(net, num_simulations=num_simulations)
    states, mcts_probs, players = [], [], []
    move_count = 0

    while not game.is_terminal():
        if move_count >= max_moves or game.board.can_claim_draw():
            print("\nGame ended due to move limit or draw rule.")
            break

        root        = mcts.search(game)
        valid_moves = game.get_valid_moves()
        visit_counts = {
            move: root.children[move].visit_count if move in root.children else 0
            for move in valid_moves
        }

        if not visit_counts:
            print("No valid moves left. Game over.")
            break

        best_move = max(visit_counts, key=visit_counts.get)

        # Record state and active player BEFORE making the move so the stored
        # FEN matches the position that produced the visit-count policy vector.
        states.append(game.board.fen())
        players.append(1 if game.board.turn == chess.WHITE else -1)

        total_visits  = sum(visit_counts.values())
        policy_vector = [0.0] * OUTPUT_DIM
        for move, count in visit_counts.items():
            policy_vector[move_to_index(move)] = count / total_visits
        mcts_probs.append(policy_vector)

        game.make_move(best_move)
        move_count += 1

    winner     = game.check_winner()
    result_str = {1: "White wins", -1: "Black wins", 0: "Draw", None: "Unfinished"}.get(winner)
    print(f"Game Over! Result: {result_str}")

    rewards = []
    for player in players:
        if winner is None or winner == 0:
            rewards.append(0)  # Neutral outcome for draws and unfinished games
        else:
            rewards.append(1 if winner == player else -1)

    return states, mcts_probs, rewards, players, winner


def train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1):
    """
    Trains the network on a batch of self-play data.
    """
    net.train()
    for epoch in range(epochs):
        total_loss = 0
        for fen, pi, reward, player in zip(states, mcts_probs, rewards, players):
            board        = chess.Board(fen)
            board_tensor = board_to_tensor(board).to(device)
            optimizer.zero_grad()
            log_pi, value = net(board_tensor)
            value_target  = torch.tensor([[reward]], dtype=torch.float32).to(device)
            value_loss    = (value - value_target).pow(2).mean()
            target_pi     = torch.tensor(pi, dtype=torch.float32).unsqueeze(0).to(device)
            policy_loss   = -torch.sum(target_pi * log_pi)
            loss          = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(states):.4f}")


if __name__ == "__main__":
    net       = ChessNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    checkpoint_path = "chess_model_checkpoint.pth"

    num_iterations = int(input("Enter number of iterations (default 50): ") or "50")
    checkpoint_freq = int(input("Enter checkpoint save frequency (default 10): ") or "10")

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        states, mcts_probs, rewards, players, _ = self_play_game(net, num_simulations=100)
        train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1)
        if (iteration + 1) % checkpoint_freq == 0 or (iteration + 1) == num_iterations:
            torch.save(
                {'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                checkpoint_path,
            )
            print(f"Checkpoint saved at iteration {iteration+1}")
