"""
Training loop: batched-GPU self-play → replay buffer → network updates.

Run with defaults:            python main.py
See all knobs:                python main.py --help
Resume happens automatically if the checkpoint file exists.
"""

import argparse
import os
import random
import time
from collections import deque

import torch

from network import ChessNet, device
from selfplay import SelfPlayEngine, SelfPlayConfig
from train import train_step

CHECKPOINT_DEFAULT = "chess_checkpoint.pth"


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero-style chess training")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--parallel-games", type=int, default=64,
                   help="concurrent self-play games = GPU batch size")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--temperature-plies", type=int, default=20)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=150_000)
    p.add_argument("--min-buffer", type=int, default=5_000,
                   help="positions required before training starts")
    p.add_argument("--train-ratio", type=float, default=1.0,
                   help="training samples consumed per new self-play position")
    p.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    p.add_argument("--checkpoint-every", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    start_iteration = 0

    if os.path.exists(args.checkpoint):
        data = torch.load(args.checkpoint, map_location=device)
        if data.get("channels", args.channels) != args.channels or \
           data.get("blocks", args.blocks) != args.blocks:
            raise SystemExit(
                f"Checkpoint {args.checkpoint} was trained with "
                f"{data.get('channels')}x{data.get('blocks')}, but you asked for "
                f"{args.channels}x{args.blocks}. Pass matching --channels/--blocks "
                f"or move the checkpoint aside to start fresh."
            )
        net.load_state_dict(data["model_state_dict"])
        optimizer.load_state_dict(data["optimizer_state_dict"])
        start_iteration = data.get("iteration", 0)
        print(f"Resumed from {args.checkpoint} at iteration {start_iteration}.")
    else:
        print("No checkpoint found. Starting from scratch.")

    config = SelfPlayConfig(
        num_simulations=args.simulations,
        parallel_games=args.parallel_games,
        temperature_plies=args.temperature_plies,
        max_game_plies=args.max_plies,
    )
    engine = SelfPlayEngine(net, device, config)
    buffer = deque(maxlen=args.buffer_size)

    for iteration in range(start_iteration, args.iterations):
        t0 = time.time()
        print(f"\nIteration {iteration + 1}/{args.iterations}")

        examples, stats = engine.play(args.games_per_iter)
        buffer.extend(examples)

        avg_plies = stats["plies"] / max(1, stats["games"])
        print(f"  games: {stats['games']} "
              f"(W {stats['white']} / D {stats['draw']} / B {stats['black']}) | "
              f"checkmates {stats['by_checkmate']}, adjudicated {stats['by_adjudication']}, "
              f"repetition {stats['by_repetition']}, fifty-move {stats['by_fifty_move']}")
        print(f"  avg plies: {avg_plies:.0f} | new positions: {len(examples)} | "
              f"buffer: {len(buffer)}")

        if len(buffer) >= args.min_buffer:
            pool = list(buffer)
            steps = max(1, round(len(examples) * args.train_ratio / args.batch_size))
            policy_losses, value_losses = [], []
            for _ in range(steps):
                batch = random.sample(pool, min(args.batch_size, len(pool)))
                pl, vl = train_step(net, optimizer, batch, device, scaler)
                policy_losses.append(pl)
                value_losses.append(vl)
            print(f"  trained {steps} steps | policy loss {sum(policy_losses)/steps:.4f} | "
                  f"value loss {sum(value_losses)/steps:.4f}")
        else:
            print(f"  buffer filling... {len(buffer)}/{args.min_buffer} before training starts")

        print(f"  iteration time: {time.time() - t0:.0f}s")

        if (iteration + 1) % args.checkpoint_every == 0 or (iteration + 1) == args.iterations:
            torch.save({
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "channels": args.channels,
                "blocks": args.blocks,
                "iteration": iteration + 1,
            }, args.checkpoint)
            print(f"  checkpoint saved: {args.checkpoint}")


if __name__ == "__main__":
    main()
