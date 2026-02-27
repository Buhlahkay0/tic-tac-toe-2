import numpy as np

class TicTacToe:
    def __init__(self):
        # Initialize a 3x3 board; 0: empty, 1: player 1, -1: player 2.
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # player 1 starts

    def clone(self):
        # Create a deep copy of the game (for simulation in MCTS).
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_valid_moves(self):
        # Return a list of valid moves as (row, col) pairs.
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, move):
        # Apply a move to the board.
        i, j = move
        if self.board[i, j] != 0:
            raise ValueError("Invalid move!")
        self.board[i, j] = self.current_player
        self.current_player *= -1  # switch players

    def check_winner(self):
        # Check rows, columns, and diagonals for a win.
        for player in [1, -1]:
            # Check rows and columns.
            for i in range(3):
                if np.all(self.board[i, :] == player):
                    return player
                if np.all(self.board[:, i] == player):
                    return player
            # Check diagonals.
            if np.all(np.diag(self.board) == player):
                return player
            if np.all(np.diag(np.fliplr(self.board)) == player):
                return player
        # If no win and no moves left, it is a draw.
        if not self.get_valid_moves():
            return 0  # draw
        return None  # game ongoing

    def is_terminal(self):
        # The game is over if there is a winner or a draw.
        return self.check_winner() is not None
