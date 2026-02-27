import chess

class ChessGame:
    def __init__(self):
        # Start with the standard chess starting position.
        self.board = chess.Board()

    def clone(self):
        new_game = ChessGame()
        # Copy the board state including the full move stack.
        new_game.board = self.board.copy(stack=True)
        return new_game

    def get_valid_moves(self):
        # Return a list of all legal moves.
        return list(self.board.legal_moves)

    def make_move(self, move):
        # Push the move onto the board.
        self.board.push(move)

    def is_terminal(self):
        # The game is over if the board reports that it is.
        return self.board.is_game_over()

    def check_winner(self):
        """
        Returns:
          1  if White wins,
         -1  if Black wins,
          0  for a draw,
          None if the game is not over.
        """
        if not self.is_terminal():
            return None
        result = self.board.result()  # e.g., "1-0", "0-1", or "1/2-1/2"
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0
