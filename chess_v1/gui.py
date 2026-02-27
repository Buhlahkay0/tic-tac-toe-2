import tkinter as tk
import threading
import chess
import torch
from chess_game import ChessGame
from mcts import MCTS
from network import ChessNet

class ChessGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chess Engine GUI")
        
        # Set up canvas dimensions.
        self.square_size = 60
        self.canvas_size = self.square_size * 8
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Initialize the game state and selection.
        self.game = ChessGame()
        self.selected_square = None  # For tracking human move selection.
        
        # Load network and initialize MCTS.
        self.net = ChessNet()
        checkpoint = "chess_model_checkpoint.pth"
        try:
            self.net.load_state_dict(torch.load(checkpoint))
            print("Loaded checkpoint from", checkpoint)
        except Exception as e:
            print("No checkpoint found, using untrained network.", e)
        self.net.eval()
        
        # Use 100 simulations for the AI move selection.
        self.mcts_simulations = 100
        self.mcts = MCTS(self.net, num_simulations=self.mcts_simulations)
        
        # Draw the initial board.
        self.draw_board()
        
        # Bind mouse clicks to the canvas.
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Start checking if it's AI's turn.
        self.master.after(100, self.check_ai_turn)

    def draw_board(self):
        self.canvas.delete("all")
        light_color = "#F0D9B5"
        dark_color = "#B58863"
        
        # Draw squares.
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = light_color if (rank + file) % 2 == 0 else dark_color
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        
        # Draw coordinate labels.
        for file in range(8):
            letter = chr(ord('a') + file)
            x = file * self.square_size + self.square_size / 2
            y = self.canvas_size - 10
            self.canvas.create_text(x, y, text=letter, font=("Arial", 10))
        for rank in range(8):
            number = str(8 - rank)
            x = 10
            y = rank * self.square_size + self.square_size / 2
            self.canvas.create_text(x, y, text=number, font=("Arial", 10))
        
        # Draw pieces.
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        for square in chess.SQUARES:
            piece = self.game.board.piece_at(square)
            if piece is not None:
                file = chess.square_file(square)
                rank = 7 - chess.square_rank(square)  # Top row = rank 8.
                x = file * self.square_size + self.square_size / 2
                y = rank * self.square_size + self.square_size / 2
                self.canvas.create_text(x, y, text=piece_symbols[piece.symbol()], font=("Arial", 32))
        
        # If a square is selected, highlight it.
        if self.selected_square is not None:
            file = chess.square_file(self.selected_square)
            rank = 7 - chess.square_rank(self.selected_square)
            x1 = file * self.square_size
            y1 = rank * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3)

    def on_click(self, event):
        # Translate click coordinates to board square.
        file = event.x // self.square_size
        rank = event.y // self.square_size
        square = chess.square(file, 7 - rank)
        
        # If no square is selected, select a square if it has a piece belonging to human (assumed White).
        if self.selected_square is None:
            piece = self.game.board.piece_at(square)
            if piece is not None and piece.color == chess.WHITE:
                self.selected_square = square
        else:
            # Attempt to make a move from the selected square to the clicked square.
            move = chess.Move(self.selected_square, square)
            if move in self.game.board.legal_moves:
                self.game.make_move(move)
                self.selected_square = None
                self.draw_board()
                # After a human move, if the game isn't over, schedule the AI move.
                if not self.game.board.is_game_over():
                    self.master.after(500, self.ai_move)
            else:
                # If move is invalid, clear selection.
                self.selected_square = None
        self.draw_board()

    def ai_move(self):
        def run_ai():
            # Run MCTS search (this can take time, so we run it in a separate thread).
            root = self.mcts.search(self.game)
            valid_moves = list(self.game.board.legal_moves)
            if valid_moves:
                visit_counts = {
                    move: root.children[move].visit_count if move in root.children else 0
                    for move in valid_moves
                }
                best_move = max(visit_counts, key=visit_counts.get)
                self.game.make_move(best_move)
            # Update the board on the main thread.
            self.master.after(100, self.draw_board)
            self.master.after(100, self.check_ai_turn)
        threading.Thread(target=run_ai).start()

    def check_ai_turn(self):
        # If it's AI's turn (assuming human plays White), initiate AI move.
        if self.game.board.turn == chess.BLACK and not self.game.board.is_game_over():
            self.ai_move()
        else:
            self.draw_board()

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()
