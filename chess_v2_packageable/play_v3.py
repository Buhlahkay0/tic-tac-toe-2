import os
import sys

import chess
import torch

from mcts import MCTS
from network import ChessNet, load_checkpoint, device


def print_board_with_coords(board, human_is_white):
    """
    Prints the board with coordinate labels, oriented for the human player.
    Pieces are shown using Unicode chess symbols; empty squares as dots.
    """
    if human_is_white:
        ranks = list(range(8, 0, -1))
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    else:
        ranks = list(range(1, 9))
        files = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']

    piece_symbols = {
        'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
        'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔'
    }

    board_str = ""
    for rank in ranks:
        row_str = ""
        for file in files:
            square = chess.parse_square(file + str(rank))
            piece = board.piece_at(square)
            row_str += (". " if piece is None else piece_symbols[piece.symbol()] + " ")
        board_str += str(rank) + " " + row_str + "\n"
    board_str += "  " + " ".join(files) + "\n"
    print(board_str)


def human_vs_ai(net, num_simulations=200, human_color="white"):
    board = chess.Board()
    mcts = MCTS()
    human_is_white = (human_color.lower() == "white")

    while not board.is_game_over(claim_draw=True):
        print_board_with_coords(board, human_is_white)
        print(f"Current turn: {'White' if board.turn == chess.WHITE else 'Black'}\n")

        if (board.turn == chess.WHITE) == human_is_white:
            move_input = input("Enter your move in UCI format (e.g., e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_input)
            except ValueError:
                print("Invalid move format. Please try again.\n")
                continue
            if move not in board.legal_moves:
                print("Illegal move. Please try again.\n")
                continue
            board.push(move)
        else:
            print("AI is thinking...\n")
            root = mcts.search(board, net, num_simulations, device)
            best_move = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
            print("AI plays:", best_move.uci(), "\n")
            board.push(best_move)

    print_board_with_coords(board, human_is_white)
    print("Game over!")
    print("Result:", board.result(claim_draw=True))


if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint = os.path.join(base_path, "chess_checkpoint.pth")

    try:
        net, _ = load_checkpoint(checkpoint)
        print("Model checkpoint loaded from", checkpoint)
    except FileNotFoundError:
        print("No checkpoint found. Using an untrained model.")
        net = ChessNet().to(device)
    net.eval()

    human_color = input("Do you want to play as White or Black? ").strip().lower()
    if human_color not in ["white", "black"]:
        print("Invalid choice. Defaulting to White.")
        human_color = "white"

    try:
        sims_input = input("Enter the number of simulations for AI (default 400): ").strip()
        num_simulations = int(sims_input) if sims_input else 400
    except ValueError:
        print("Invalid input. Defaulting to 400 simulations.")
        num_simulations = 400

    human_vs_ai(net, num_simulations=num_simulations, human_color=human_color)
