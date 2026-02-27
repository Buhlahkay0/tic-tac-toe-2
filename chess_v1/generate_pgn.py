import os
import torch
import chess
import chess.pgn
from chess_game import ChessGame
from mcts import MCTS
from network import ChessNet

def generate_self_play_pgn(net, num_simulations=100, save_path="self_play_game.pgn"):
    """
    Runs a self-play game between two AI instances and saves it in PGN format.
    Args:
        net (ChessNet): The trained neural network.
        num_simulations (int): Number of MCTS simulations per move.
        save_path (str): File path to save the PGN file.
    """
    game = ChessGame()
    mcts = MCTS(net, num_simulations=num_simulations)

    # Create PGN game object
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Self-Play Game"
    pgn_game.headers["White"] = "AI"
    pgn_game.headers["Black"] = "AI"
    
    node = pgn_game

    print("\nStarting AI vs AI self-play game...")
    while not game.is_terminal():
        print(game.board)
        
        root = mcts.search(game)
        valid_moves = game.get_valid_moves()
        visit_counts = {move: root.children[move].visit_count if move in root.children else 0 for move in valid_moves}

        if not visit_counts:
            print("No valid moves left. Game over.")
            break
        
        best_move = max(visit_counts, key=visit_counts.get)
        print(f"\nAI plays: {game.board.san(best_move)}\n")
        
        game.make_move(best_move)
        node = node.add_variation(best_move)  # Add move to PGN

    print("\nGame Over! Result:", game.board.result())

    # Save PGN file
    with open(save_path, "w") as pgn_file:
        pgn_file.write(str(pgn_game))
    
    print(f"\nPGN saved to {save_path}. You can upload it to Lichess to view the game.")

if __name__ == "__main__":
    # Load the model on CPU
    net = ChessNet().to("cpu")  # Force it to run on CPU

    # Determine base directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint = os.path.join(base_path, "chess_model_checkpoint_3000.pth")

    # Load model checkpoint if available
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        net.load_state_dict(checkpoint_data["model_state_dict"])
        print("Model checkpoint loaded from", checkpoint)
    except Exception as e:
        print("No checkpoint found. Using an untrained model.", e)

    net.eval()  # Set to evaluation mode

    # User input for simulations
    try:
        num_simulations = int(input("Enter the number of simulations per move (default 100): ") or "100")
    except ValueError:
        print("Invalid input. Defaulting to 100 simulations.")
        num_simulations = 100

    # Run self-play and generate PGN
    generate_self_play_pgn(net, num_simulations=num_simulations)
