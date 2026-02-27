import numpy as np
from tictactoe import TicTacToe
from mcts import MCTS
from network import TicTacToeNet
import torch

def print_board(board):
    """
    Print the board in a human-readable format.
    'X' represents 1, 'O' represents -1, and '.' represents an empty cell.
    """
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for row in board:
        print(" ".join([symbols[cell] for cell in row]))
    print()  # extra line for spacing

def play_human_vs_ai(net, num_simulations=50, human_player=1):
    """
    Lets a human play against the AI.
    
    Parameters:
    - net: the trained TicTacToeNet.
    - num_simulations: number of MCTS simulations for the AI's move.
    - human_player: which side the human will play (1 for 'X' or -1 for 'O').
    """
    game = TicTacToe()
    mcts = MCTS(net, num_simulations=num_simulations)

    print("Starting a new game of Tic Tac Toe!")
    print("You are '{}'.".format('X' if human_player == 1 else 'O'))
    
    while not game.is_terminal():
        print_board(game.board)
        
        if game.current_player == human_player:
            valid_moves = game.get_valid_moves()
            print("Your turn. Valid moves (row, col):", valid_moves)
            move_input = input("Enter your move as row,col (e.g., 0,2): ")
            try:
                move = tuple(map(int, move_input.strip().split(',')))
            except Exception as e:
                print("Invalid input format. Please try again.")
                continue
            if move not in valid_moves:
                print("That move is not valid. Try one of:", valid_moves)
                continue
            game.make_move(move)
        else:
            print("AI is thinking...")
            root = mcts.search(game)
            # Choose the move with the highest visit count.
            visit_counts = np.array([
                root.children[(i, j)].visit_count if (i, j) in root.children else 0 
                for i in range(3) for j in range(3)
            ])
            move_index = np.argmax(visit_counts)
            move = (move_index // 3, move_index % 3)
            print("AI plays:", move)
            game.make_move(move)
    
    print_board(game.board)
    winner = game.check_winner()
    if winner == 0:
        print("The game ended in a draw!")
    elif winner == human_player:
        print("Congratulations! You win!")
    else:
        print("AI wins! Better luck next time.")

if __name__ == "__main__":
    # Load or initialize your network.
    net = TicTacToeNet()
    
    # If you have a pre-trained model, you might load its state dict:
    # net.load_state_dict(torch.load("path_to_trained_model.pth"))
    
    # Set the network to evaluation mode.
    net.eval()
    
    # Start the game: You can set human_player to 1 (X) or -1 (O)
    play_human_vs_ai(net, num_simulations=1000, human_player=1)
