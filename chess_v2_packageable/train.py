import random
import torch
import torch.nn.functional as F
import chess
import numpy as np
from chess_game import ChessGame
from mcts import MCTS, MCTSNode, move_to_index
from network import ChessNet, board_to_tensor, boards_to_batch, OUTPUT_DIM, device
from play_v3 import print_board_with_coords

TEMPERATURE_MOVES = 15  # use temperature sampling for the first N moves of each game


def _select_move_with_temperature(visit_counts):
    """Sample a move proportionally to visit counts (exploration)."""
    moves  = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    probs  = counts / counts.sum()
    return moves[np.random.choice(len(moves), p=probs)]


def self_play_game(net, num_simulations=100, max_moves=100, verbose=True, eval_batch_size=16):
    """
    Runs a single self-play game using MCTS for move selection.
    States and player labels are recorded BEFORE each move so they correspond
    to the position that generated the policy vector.
    Returns training data, the final winner, and the final board FEN.
    Set verbose=False to suppress all output.
    """
    game  = ChessGame()
    mcts  = MCTS(net, num_simulations=num_simulations, eval_batch_size=eval_batch_size)
    states, mcts_probs, players = [], [], []
    move_count = 0

    while not game.is_terminal():
        if move_count >= max_moves:
            if verbose:
                print("Game ended: move limit reached.")
            break

        root        = mcts.search(game, add_noise=True)
        valid_moves = game.get_valid_moves()
        visit_counts = {
            move: root.children[move].visit_count if move in root.children else 0
            for move in valid_moves
        }

        if not visit_counts:
            if verbose:
                print("No valid moves left. Game over.")
            break

        # Filter out moves that would immediately allow a draw claim, unless
        # every legal move does so (forced draw).
        non_draw_moves = {}
        for move, count in visit_counts.items():
            test_game = game.clone()
            test_game.make_move(move)
            if not test_game.board.can_claim_draw():
                non_draw_moves[move] = count
        if non_draw_moves:
            visit_counts = non_draw_moves
        else:
            if verbose:
                print("Game ended: draw unavoidable.")
            break

        # Temperature sampling for early moves, greedy afterwards.
        if move_count < TEMPERATURE_MOVES:
            best_move = _select_move_with_temperature(visit_counts)
        else:
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

    winner    = game.check_winner()
    final_fen = game.board.fen()
    if verbose:
        result_str = {1: "White wins", -1: "Black wins", 0: "Draw", None: "Unfinished"}.get(winner)
        print(f"Game Over! Result: {result_str}")

    rewards = []
    for player in players:
        if winner is None or winner == 0:
            rewards.append(0)
        else:
            rewards.append(1 if winner == player else -1)

    return states, mcts_probs, rewards, players, winner, final_fen


def batched_self_play(net, num_games=4, num_simulations=100, max_moves=150,
                      eval_batch_size=16, verbose=True):
    """
    Runs num_games self-play games simultaneously in one process, sharing a single
    GPU network.  All pending leaves from all active games are pooled into one
    batched GPU call per MCTS inner-loop iteration.

    Returns a list of (states, mcts_probs, rewards, players, winner, final_fen)
    tuples, one per game, in the order they were created.
    """
    mcts = MCTS(net, num_simulations=num_simulations, eval_batch_size=eval_batch_size)

    # Per-game state
    games        = [ChessGame()             for _ in range(num_games)]
    roots        = [None]                   * num_games   # filled after first expand
    sims_done    = [0]                      * num_games   # simulations completed for current move
    move_counts  = [0]                      * num_games
    states_list  = [[]                      for _ in range(num_games)]
    probs_list   = [[]                      for _ in range(num_games)]
    players_list = [[]                      for _ in range(num_games)]
    done         = [False]                  * num_games
    results      = [None]                   * num_games   # final tuple per game

    # Initialise roots: expand each root once with a single batched call
    initial_nodes = []
    for i in range(num_games):
        root = MCTSNode(games[i])
        roots[i] = root
        initial_nodes.append(root)
    mcts._batch_expand(initial_nodes)
    for i in range(num_games):
        if roots[i].children:
            mcts._add_dirichlet_noise(roots[i])

    # Pending-leaf state for each game within the current MCTS inner iteration
    # Each entry is (leaves, paths, to_expand) or None when the game is idle
    pending = [None] * num_games

    def _finish_game(i):
        """Record results for game i and mark it done."""
        game = games[i]
        winner    = game.check_winner()
        final_fen = game.board.fen()
        if verbose:
            result_str = {1: "White wins", -1: "Black wins", 0: "Draw",
                          None: "Unfinished"}.get(winner)
            print(f"  [Game {i+1}] Over — {result_str}")
        rewards_i = []
        for player in players_list[i]:
            if winner is None or winner == 0:
                rewards_i.append(0)
            else:
                rewards_i.append(1 if winner == player else -1)
        results[i] = (states_list[i], probs_list[i], rewards_i, players_list[i],
                      winner, final_fen)
        done[i] = True

    def _record_and_advance(i):
        """
        After simulations for move i are done: pick the move, record training
        data, advance the game.  Returns False if the game ended, True otherwise.
        """
        game  = games[i]
        root  = roots[i]
        valid_moves = game.get_valid_moves()
        visit_counts = {
            move: root.children[move].visit_count if move in root.children else 0
            for move in valid_moves
        }

        if not visit_counts:
            if verbose:
                print(f"  [Game {i+1}] No valid moves. Ending game.")
            _finish_game(i)
            return False

        # Filter draw-inducing moves unless all moves cause draws
        non_draw_moves = {}
        for move, count in visit_counts.items():
            test_game = game.clone()
            test_game.make_move(move)
            if not test_game.board.can_claim_draw():
                non_draw_moves[move] = count
        if non_draw_moves:
            visit_counts = non_draw_moves
        else:
            if verbose:
                print(f"  [Game {i+1}] Draw unavoidable. Ending game.")
            _finish_game(i)
            return False

        # Temperature / greedy move selection
        mc = move_counts[i]
        if mc < TEMPERATURE_MOVES:
            best_move = _select_move_with_temperature(visit_counts)
        else:
            best_move = max(visit_counts, key=visit_counts.get)

        # Record training data for this position
        states_list[i].append(game.board.fen())
        players_list[i].append(1 if game.board.turn == chess.WHITE else -1)
        total_visits  = sum(visit_counts.values())
        policy_vector = [0.0] * OUTPUT_DIM
        for move, count in visit_counts.items():
            policy_vector[move_to_index(move)] = count / total_visits
        probs_list[i].append(policy_vector)

        game.make_move(best_move)
        move_counts[i] += 1

        # Check terminal / move limit
        if game.is_terminal() or move_counts[i] >= max_moves:
            if move_counts[i] >= max_moves and not game.is_terminal() and verbose:
                print(f"  [Game {i+1}] Move limit reached.")
            _finish_game(i)
            return False

        # Start new MCTS root for the next position
        new_root = MCTSNode(game)
        roots[i] = new_root
        sims_done[i] = 0
        return True   # game still active; caller must expand new root

    # ---------- main loop ----------
    while not all(done):
        # Gather the set of active games that still need more simulations this move
        active = [i for i in range(num_games) if not done[i]]

        # Collect leaves for one inner-loop batch from every active game
        all_tensors = []   # board tensor for each node needing GPU eval

        for i in active:
            batch_n = min(eval_batch_size, num_simulations - sims_done[i])
            leaves, paths, to_expand, board_tensors = mcts.collect_leaves(
                roots[i], batch_n
            )
            pending[i] = (leaves, paths, to_expand)
            if board_tensors.shape[0] > 0:
                all_tensors.append(board_tensors)

        # Single batched GPU forward pass over ALL games' leaves.
        # all_tensors is a list of (N_i, 12, 8, 8) CPU tensors, one per game.
        # One cat + one .to(device) covers all games' leaves in a single transfer.
        if all_tensors:
            batch = torch.cat(all_tensors, dim=0).to(device)
            with torch.no_grad():
                log_policies, values_t = net(batch)
            policies_np = log_policies.exp().cpu().numpy()
            values_np   = values_t.cpu().numpy().flatten()
        else:
            policies_np = np.zeros((0,))
            values_np   = np.zeros((0,))

        # Distribute results back to each game and run process_leaves
        idx = 0
        for i in active:
            leaves, paths, to_expand = pending[i]
            n = len(to_expand)
            game_policies = policies_np[idx:idx + n] if n > 0 else np.zeros((0,))
            game_values   = values_np[idx:idx + n]   if n > 0 else np.zeros((0,))
            idx += n

            mcts.process_leaves(roots[i], leaves, paths, to_expand,
                                 game_policies, game_values)
            batch_n = len(leaves)
            sims_done[i] += batch_n

        # Check whether any game has finished its simulations for the current move
        # Collect games that need a new root expanded so we can batch that too
        needs_new_root_expand = []
        for i in active:
            if sims_done[i] >= num_simulations:
                still_active = _record_and_advance(i)
                if still_active:
                    needs_new_root_expand.append(i)

        # Batch-expand all newly created roots at once
        if needs_new_root_expand:
            new_root_nodes = [roots[i] for i in needs_new_root_expand]
            mcts._batch_expand(new_root_nodes)
            for i in needs_new_root_expand:
                if roots[i].children:
                    mcts._add_dirichlet_noise(roots[i])

    return results


def train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1, scaler=None):
    """
    Trains the network on a batch of self-play data.
    All positions are evaluated in a single forward pass per epoch, which keeps
    the GPU fully occupied instead of issuing hundreds of tiny batch-size-1 calls.
    """
    net.train()

    # Build all tensors once on CPU, then move to device in one transfer.
    boards      = [chess.Board(fen) for fen in states]
    board_batch = boards_to_batch(boards).to(device)                          # (N, 12, 8, 8)
    target_pi   = torch.tensor(mcts_probs, dtype=torch.float32, device=device)  # (N, OUTPUT_DIM)
    target_v    = torch.tensor(rewards,    dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            log_pi, value = net(board_batch)                     # single forward pass
            value_loss  = F.mse_loss(value, target_v)
            policy_loss = -(target_pi * log_pi).sum(dim=1).mean()
            loss        = value_loss + policy_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    net       = ChessNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    checkpoint_path = "chess_model_checkpoint.pth"

    num_iterations = int(input("Enter number of iterations (default 50): ") or "50")
    checkpoint_freq = int(input("Enter checkpoint save frequency (default 10): ") or "10")

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        states, mcts_probs, rewards, players, _, _fen = self_play_game(net, num_simulations=100)
        train_network(net, optimizer, states, mcts_probs, rewards, players, epochs=1)
        if (iteration + 1) % checkpoint_freq == 0 or (iteration + 1) == num_iterations:
            torch.save(
                {'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                checkpoint_path,
            )
            print(f"Checkpoint saved at iteration {iteration+1}")
