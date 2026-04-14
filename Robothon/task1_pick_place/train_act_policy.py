"""
ACT Policy Training for Task 1 – Pick and Place
Physical AI Hackathon 2026

Trains an Action Chunking Transformer (ACT) policy on the collected
demonstrations using the LeRobot library.

Two modes:
  1. LeRobot native  – uses `lerobot.scripts.train` with a config
  2. Standalone ACT  – minimal PyTorch implementation for environments
                       where the full LeRobot library is not available

Usage:
    # Mode 1: LeRobot native (recommended if lerobot is installed)
    python train_act_policy.py --mode lerobot --data_dir demo_data

    # Mode 2: Standalone training
    python train_act_policy.py --mode standalone --data_dir demo_data

    # Evaluate a checkpoint
    python train_act_policy.py --mode eval --checkpoint checkpoints/best.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lerobot_env import SO101PickPlaceEnv, ACTION_DIM, OBS_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PickPlaceDemoDataset(Dataset):
    """
    Loads demonstration data saved by collect_demos.py.

    Each sample is a (observation_chunk, action_chunk) pair of length
    `chunk_size` starting at a random offset in any episode.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 20,
        obs_key: str = "observation.state",
    ):
        self.chunk_size = chunk_size
        self.obs_key    = obs_key

        data_dir = Path(data_dir)
        meta_path = data_dir / "meta_data.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"meta_data.json not found in {data_dir}. "
                "Run collect_demos.py first."
            )

        with open(meta_path) as f:
            meta = json.load(f)

        # Load normalisation stats
        stats_path = data_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.obs_mean = torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32)
            self.obs_std  = torch.tensor(stats["observation.state"]["std"],  dtype=torch.float32)
            self.act_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32)
            self.act_std  = torch.tensor(stats["action"]["std"],  dtype=torch.float32)
        else:
            self.obs_mean = torch.zeros(OBS_DIM)
            self.obs_std  = torch.ones(OBS_DIM)
            self.act_mean = torch.zeros(ACTION_DIM)
            self.act_std  = torch.ones(ACTION_DIM)

        # Build index: list of (file_path, start_idx) tuples
        self.index = []
        for ep in meta["episodes"]:
            fpath = ep["file"]
            n     = ep["num_frames"]
            for start in range(0, n - chunk_size):
                self.index.append((fpath, start))

        if len(self.index) == 0:
            raise ValueError(
                f"Dataset has no valid windows. Check chunk_size ({chunk_size}) "
                f"vs episode lengths."
            )

        print(f"Dataset: {len(self.index)} windows from "
              f"{len(meta['episodes'])} episodes")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, start = self.index[idx]
        d = np.load(fpath)
        end = start + self.chunk_size

        obs  = torch.from_numpy(d[self.obs_key][start:end]).float()
        act  = torch.from_numpy(d["action"][start:end]).float()

        # Normalise
        obs_norm = (obs - self.obs_mean) / self.obs_std
        act_norm = (act - self.act_mean) / self.act_std

        # Return first obs as context + full action chunk
        return obs_norm[0], act_norm   # (OBS_DIM,),  (chunk_size, ACTION_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone ACT Model (minimal PyTorch implementation)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class ACTPolicy(nn.Module):
    """
    Action Chunking Transformer (ACT) for pick-and-place.

    Architecture:
      Encoder: MLP obs → token
      Transformer: 4-layer causal transformer
      Decoder: predict chunk_size actions
    """

    def __init__(
        self,
        obs_dim:    int   = OBS_DIM,
        action_dim: int   = ACTION_DIM,
        chunk_size: int   = 20,
        d_model:    int   = 256,
        n_heads:    int   = 8,
        n_layers:   int   = 4,
        d_ff:       int   = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.chunk_size  = chunk_size
        self.action_dim  = action_dim
        self.d_model     = d_model

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable query tokens for action prediction
        self.action_queries = nn.Embedding(chunk_size, d_model)

        # Transformer encoder (context)
        self.transformer_encoder = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout)
              for _ in range(n_layers)]
        )

        # Transformer decoder (predict actions)
        self.transformer_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout,
                batch_first=True, activation="gelu"
            )
            for _ in range(n_layers)
        ])

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, action_dim),
            nn.Tanh(),   # actions are normalised to [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)  current observation
        Returns:
            actions: (B, chunk_size, action_dim)
        """
        B = obs.shape[0]

        # Encode observation into memory token
        memory = self.obs_encoder(obs).unsqueeze(1)  # (B, 1, d_model)

        # Expand memory through encoder
        for layer in self.transformer_encoder:
            memory = layer(memory)

        # Action query tokens
        idx = torch.arange(self.chunk_size, device=obs.device)
        queries = self.action_queries(idx).unsqueeze(0).expand(B, -1, -1)  # (B, T, d)

        # Decoder
        out = queries
        for layer in self.transformer_decoder:
            out = layer(out, memory)

        # Predict actions
        actions = self.action_head(out)  # (B, chunk_size, action_dim)
        return actions

    def predict(self, obs: torch.Tensor) -> np.ndarray:
        """Inference-only wrapper. Returns numpy array."""
        self.eval()
        with torch.no_grad():
            actions = self.forward(obs.unsqueeze(0))
        return actions.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_standalone(args):
    """Standalone training without LeRobot library."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset
    dataset = PickPlaceDemoDataset(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Model
    model = ACTPolicy(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        chunk_size=args.chunk_size,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimiser with cosine LR schedule
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss  = float("inf")
    train_log  = []

    print(f"\nTraining for {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_obs, batch_act in dataloader:
            batch_obs = batch_obs.to(device)   # (B, OBS_DIM)
            batch_act = batch_act.to(device)   # (B, chunk_size, ACTION_DIM)

            # Forward
            pred_act = model(batch_obs)

            # L1 + L2 mixed loss (L1 for robustness, L2 for smoothness)
            loss = F.l1_loss(pred_act, batch_act) * 0.5 + \
                   F.mse_loss(pred_act, batch_act) * 0.5

            # Backward
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item() * batch_obs.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        train_log.append({"epoch": epoch, "loss": avg_loss,
                          "lr": scheduler.get_last_lr()[0]})

        if epoch % args.log_every == 0 or epoch == 1:
            dt = time.time() - t0
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"loss={avg_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"dt={dt:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "loss":        avg_loss,
                "config": {
                    "obs_dim":    OBS_DIM,
                    "action_dim": ACTION_DIM,
                    "chunk_size": args.chunk_size,
                    "d_model":    args.d_model,
                    "n_heads":    args.n_heads,
                    "n_layers":   args.n_layers,
                },
                "stats": {
                    "obs_mean": dataset.obs_mean.tolist(),
                    "obs_std":  dataset.obs_std.tolist(),
                    "act_mean": dataset.act_mean.tolist(),
                    "act_std":  dataset.act_std.tolist(),
                },
            }, ckpt_dir / "best.pt")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "loss": avg_loss},
                ckpt_dir / f"epoch_{epoch:04d}.pt"
            )

    # Save training log
    with open(ckpt_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.5f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    return model, dataset


# ─────────────────────────────────────────────────────────────────────────────
# LeRobot native training
# ─────────────────────────────────────────────────────────────────────────────

def train_lerobot(args):
    """
    Generate a LeRobot training config and invoke lerobot.scripts.train.
    Requires: pip install lerobot
    """
    try:
        import lerobot
    except ImportError:
        print("LeRobot not installed. Run: pip install lerobot")
        print("Falling back to standalone training …")
        return train_standalone(args)

    from omegaconf import OmegaConf

    # Build config for ACT policy on our custom environment
    cfg = {
        "policy": {
            "_target_": "lerobot.common.policies.act.modeling_act.ACTPolicy",
            "input_shapes": {
                "observation.state": [OBS_DIM],
            },
            "output_shapes": {
                "action": [ACTION_DIM],
            },
            "input_normalization_modes": {
                "observation.state": "mean_std",
            },
            "output_normalization_modes": {
                "action": "mean_std",
            },
            "chunk_size":   args.chunk_size,
            "n_action_steps": args.chunk_size,
            "n_obs_steps":  1,
            "dim_model":    args.d_model,
            "n_heads":      args.n_heads,
            "n_encoder_layers": args.n_layers,
            "n_decoder_layers": args.n_layers,
        },
        "dataset": {
            "repo_id": args.data_dir,
            "episodes": None,
        },
        "training": {
            "offline_steps":        args.epochs * 1000,
            "batch_size":           args.batch_size,
            "lr":                   args.lr,
            "lr_scheduler":         "cosine",
            "lr_warmup_steps":      500,
            "grad_clip_norm":       1.0,
            "log_freq":             args.log_every,
            "save_checkpoint":      True,
            "save_freq":            args.save_every,
        },
        "eval": {
            "n_episodes":     5,
            "batch_size":     1,
        },
        "output_dir":   args.checkpoint_dir,
    }

    cfg_path = Path(args.checkpoint_dir) / "lerobot_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), cfg_path)
    print(f"LeRobot config written to: {cfg_path}")
    print("To train, run:")
    print(f"  python -m lerobot.scripts.train --config-path {cfg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    """Run a trained ACT policy in the environment."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["config"]
    stats = ckpt.get("stats", {})

    model = ACTPolicy(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, loss {ckpt['loss']:.5f})")

    # Normalisation stats
    obs_mean = torch.tensor(stats.get("obs_mean", [0]*OBS_DIM), dtype=torch.float32).to(device)
    obs_std  = torch.tensor(stats.get("obs_std",  [1]*OBS_DIM), dtype=torch.float32).to(device)
    act_mean = torch.tensor(stats.get("act_mean", [0]*ACTION_DIM), dtype=torch.float32).to(device)
    act_std  = torch.tensor(stats.get("act_std",  [1]*ACTION_DIM), dtype=torch.float32).to(device)

    env = SO101PickPlaceEnv(
        render_mode="human",
        max_episode_steps=400,
        randomise_object=True,
    )

    results = []
    for ep in range(args.eval_episodes):
        obs, info = env.reset(seed=ep)
        action_buffer = []
        buf_idx       = 0
        total_reward  = 0.0

        for _ in range(400):
            if buf_idx >= len(action_buffer):
                # Re-plan: get new action chunk
                obs_t  = torch.from_numpy(obs).float().to(device)
                obs_t  = (obs_t - obs_mean) / obs_std
                with torch.no_grad():
                    pred = model(obs_t.unsqueeze(0)).squeeze(0)  # (chunk, action_dim)
                # De-normalise
                pred = pred * act_std + act_mean
                action_buffer = pred.cpu().numpy()
                buf_idx = 0

            action = action_buffer[buf_idx]
            buf_idx += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        results.append({
            "episode":          ep,
            "success":          info["success"],
            "placement_error_m": info["placement_error_m"],
            "total_reward":      total_reward,
        })
        print(f"  ep {ep+1:3d}  "
              f"{'✓' if info['success'] else '✗'}  "
              f"err={info['placement_error_m']*100:.1f}cm  "
              f"rew={total_reward:.1f}")

    env.close()

    n_ok = sum(r["success"] for r in results)
    print(f"\nEval results: {n_ok}/{len(results)} success "
          f"({n_ok/len(results)*100:.0f}%)")
    avg_err = np.mean([r["placement_error_m"] for r in results]) * 100
    print(f"Avg placement error: {avg_err:.1f} cm")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate ACT policy for SO101 pick-and-place"
    )
    parser.add_argument("--mode", choices=["standalone", "lerobot", "eval"],
                        default="standalone",
                        help="Training mode")
    # Data
    parser.add_argument("--data_dir",       type=str, default="demo_data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint",     type=str, default="checkpoints/best.pt",
                        help="Path to checkpoint for eval mode")
    # Model
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--d_model",    type=int, default=256)
    parser.add_argument("--n_heads",    type=int, default=8)
    parser.add_argument("--n_layers",   type=int, default=4)
    # Training
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--log_every",  type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    # Eval
    parser.add_argument("--eval_episodes", type=int, default=10)

    args = parser.parse_args()

    print("=" * 60)
    print("  Task 1 – ACT Policy Training")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    if args.mode == "standalone":
        train_standalone(args)
    elif args.mode == "lerobot":
        train_lerobot(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
