import chess
import torch
from chess_game import ChessGame
from mcts import MCTS
from network import ChessNet
import sys

def print_board(board):
    # Use python-chess built-in board drawing.
    print(board)
    print()

def human_vs_ai(net, num_simulations=100, human_color="white"):
    game = ChessGame()
    mcts = MCTS(net, num_simulations=num_simulations)
    
    # Determine whether the human is playing White.
    human_is_white = (human_color.lower() == "white")
    
    while not game.is_terminal():
        print_board(game.board)
        
        if game.board.turn == chess.WHITE:
            turn_str = "White"
        else:
            turn_str = "Black"
        print(f"Current turn: {turn_str}")
        
        # Decide whose turn it is.
        if (human_is_white and game.board.turn == chess.WHITE) or (not human_is_white and game.board.turn == chess.BLACK):
            # Human move.
            move_input = input("Enter your move in UCI format (e.g., e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_input)
                if move not in game.board.legal_moves:
                    print("Illegal move. Please try again.\n")
                    continue
            except Exception as e:
                print("Invalid move format. Please try again.\n")
                continue
            game.make_move(move)
        else:
            # AI move.
            print("AI is thinking...\n")
            root = mcts.search(game)
            valid_moves = game.get_valid_moves()
            visit_counts = {}
            for move in valid_moves:
                visit_counts[move] = root.children[move].visit_count if move in root.children else 0
            if not visit_counts:
                print("AI found no valid moves. Exiting.")
                break
            best_move = max(visit_counts, key=visit_counts.get)
            print("AI plays:", best_move.uci(), "\n")
            game.make_move(best_move)
    
    print_board(game.board)
    print("Game over!")
    print("Result:", game.board.result())

if __name__ == "__main__":
    net = ChessNet()
    checkpoint = "chess_model_checkpoint.pth"
    try:
        net.load_state_dict(torch.load(checkpoint))
        print("Model checkpoint loaded from", checkpoint)
    except Exception as e:
        print("No checkpoint found. Using an untrained model.", e)
    
    net.eval()
    human_color = input("Do you want to play as White or Black? ").strip().lower()
    if human_color not in ["white", "black"]:
        print("Invalid choice. Defaulting to White.")
        human_color = "white"
    human_vs_ai(net, num_simulations=100, human_color=human_color)
