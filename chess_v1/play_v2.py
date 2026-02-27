import chess
import torch
import sys
import random
from chess_game import ChessGame
from mcts import MCTS
from network import ChessNet

def print_board_with_coords(board, human_is_white):
    """
    Prints the board with coordinate labels.
    
    If human_is_white is True:
      - Ranks are displayed from 8 (top) to 1 (bottom).
      - Files are shown from a to h (left to right).
    If human_is_white is False:
      - The board is flipped: ranks 1 (top) to 8 (bottom) and files h to a (left to right).
    """
    # Define rank order and file order based on orientation.
    if human_is_white:
        ranks = range(8, 0, -1)
        files = ['a','b','c','d','e','f','g','h']
    else:
        ranks = range(1, 9)
        files = ['h','g','f','e','d','c','b','a']
    
    board_str = ""
    for rank in ranks:
        row_str = ""
        for file in files:
            square = chess.parse_square(file + str(rank))
            piece = board.piece_at(square)
            if piece is None:
                row_str += ". "
            else:
                row_str += piece.symbol() + " "
        board_str += str(rank) + " " + row_str + "\n"
    # Print file labels at bottom.
    board_str += "  " + " ".join(files) + "\n"
    print(board_str)

def human_vs_ai(net, num_simulations=100, human_color="white"):
    game = ChessGame()
    mcts = MCTS(net, num_simulations=num_simulations)
    
    human_is_white = (human_color.lower() == "white")
    
    while not game.is_terminal():
        print_board_with_coords(game.board, human_is_white)
        
        current_turn = "White" if game.board.turn == chess.WHITE else "Black"
        print(f"Current turn: {current_turn}\n")
        
        if (human_is_white and game.board.turn == chess.WHITE) or (not human_is_white and game.board.turn == chess.BLACK):
            # Human's turn.
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
            # AI's turn.
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
    
    print_board_with_coords(game.board, human_is_white)
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
