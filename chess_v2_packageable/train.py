"""Training step over self-play examples sampled from the replay buffer."""

import contextlib

import chess
import numpy as np
import torch
import torch.nn.functional as F

from encoding import board_to_planes, OUTPUT_DIM


def encode_examples(examples, device):
    """
    Examples are (fen, policy_indices, policy_probs, value) tuples.
    Returns (planes, target_policy, target_value) tensors on `device`.
    """
    planes = np.stack([board_to_planes(chess.Board(fen)) for fen, *_ in examples])
    x = torch.from_numpy(planes).to(device)

    target_pi = torch.zeros((len(examples), OUTPUT_DIM), dtype=torch.float32)
    for i, (_, indices, probs, _) in enumerate(examples):
        target_pi[i, torch.tensor(indices)] = torch.tensor(probs, dtype=torch.float32)

    target_v = torch.tensor([e[3] for e in examples], dtype=torch.float32).unsqueeze(1)
    return x, target_pi.to(device), target_v.to(device)


def train_step(net, optimizer, examples, device, scaler=None):
    """One optimizer step on a batch of examples. Returns (policy_loss, value_loss)."""
    net.train()
    x, target_pi, target_v = encode_examples(examples, device)

    optimizer.zero_grad(set_to_none=True)
    autocast = torch.autocast("cuda") if scaler is not None else contextlib.nullcontext()
    with autocast:
        log_pi, value = net(x)
        value_loss = F.mse_loss(value, target_v)
        policy_loss = -(target_pi * log_pi).sum(dim=1).mean()
        loss = value_loss + policy_loss

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return float(policy_loss.item()), float(value_loss.item())
