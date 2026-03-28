"""
Diffusion model for human motion trajectory generation.
"""

import os
from multiprocessing import Pool
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import transforms3d as t3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────

class Config:
    # Data
    data_dir        = "retargeted"
    max_seq_len     = 128
    min_seq_len     = 16

    # Raw clip storage: pos(3) + quat(4) per joint, 1 root + 4 EE + root Z height = 36
    n_ee            = 4
    raw_dim         = 1 + (1 + n_ee) * (3 + 4)   # 36

    # Model feature dim: pos(3) + 6D(6) per joint
    feat_dim        = 1 + (1 + n_ee) * (3 + 6)   # 46

    # Input perturbation
    input_perturb_std = 0.01

    # Diffusion
    n_timesteps     = 1000
    beta_start      = 1e-4
    beta_end        = 0.02
    loss_type       = "l2"

    # Transformer
    d_model         = 512
    nhead           = 8
    num_layers      = 3
    dim_feedforward = 1024
    dropout         = 0.1

    # Training
    batch_size      = 128
    lr              = 1e-4
    weight_decay    = 1e-4
    epochs          = 500
    grad_clip       = 1.0
    val_split       = 0.1
    num_workers     = 32
    seed            = 28210

    # EMA
    ema_decay       = 0.9999

    # Checkpointing
    ckpt_dir        = "checkpoints"
    save_every      = 5
    log_every       = 1

    device          = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# ─────────────────────────────────────────
#  ROTATION UTILITIES
# ─────────────────────────────────────────

def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → 6D (first two columns, row-major)."""
    return R[:, :2].T.reshape(6)          # shape (6,)


def rot6d_to_rotmat_torch(r6d: torch.Tensor) -> torch.Tensor:
    """(..., 6) → (..., 3, 3) via Gram-Schmidt."""
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)   # (..., 3, 3)

def raw_to_feat(raw: np.ndarray) -> np.ndarray:
    """
    Convert a (T, RAW_DIM=36) quaternion clip to (T, FEAT_DIM=46) 6D-rotation clip.
    """
    T   = len(raw)
    out = np.zeros((T, cfg.feat_dim), dtype=np.float32)

    out[:, 0] = raw[:, 0]   # root Z height

    # Iterates over root + 4 EE, copying pos and converting quat -> 6D rot in one pass.
    for k in range(cfg.n_ee + 1):   # root + 4 EE
        ri = 1 + k * 7   # index into raw
        fi = 1 + k * 9   # index into feat
        out[:, fi:fi + 3] = raw[:, ri:ri + 3]
        for t in range(T):
            ee_rmat = t3d.quaternions.quat2mat(raw[t, ri + 3:ri + 7])   # (3, 3)
            out[t, fi + 3:fi + 9] = rotmat_to_6d(ee_rmat)
    return out

# ─────────────────────────────────────────
#  EMA HELPER
# ─────────────────────────────────────────

class EMA:
    """
    Maintains an exponential moving average of model weights.

    Usage:
        ema = EMA(model, decay=0.9999)
        # after each optimizer.step():
        ema.update(model)
        # to evaluate with EMA weights:
        with ema.average_parameters(model):
            val_loss = evaluate(model, ...)
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        # Deep-copy the initial parameters as shadow weights
        self.shadow = copy.deepcopy(model.state_dict())
        for v in self.shadow.values():
            v.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_sd = model.state_dict()
        for key, shadow_val in self.shadow.items():
            model_val = model_sd[key].float()
            shadow_val.copy_(shadow_val * self.decay + model_val * (1.0 - self.decay))

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = sd

    class _ContextManager:
        """Temporarily swap model weights with EMA shadow weights."""
        def __init__(self, ema: "EMA", model: nn.Module):
            self.ema   = ema
            self.model = model

        def __enter__(self):
            self._orig = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(self.ema.shadow)
            return self.model

        def __exit__(self, *args):
            self.model.load_state_dict(self._orig)

    def average_parameters(self, model: nn.Module) -> "_ContextManager":
        return self._ContextManager(self, model)

# ─────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────

EE_KEYS = ["l_foot", "r_foot", "l_hand", "r_hand"]


def load_json_clip(path: Path) -> np.ndarray | None:
    """
    Load one JSON file -> float32 array of shape (T, RAW_DIM=36).

    Root:
      - Height: Z position in world space (not delta).
      - Position delta: XYZ .
      - Orientation delta: wxyz world quaternion.

    End-effectors (root-local frame):
      - Position: R_root.T @ (ee_world - root_world)
      - Orientation: quaternion of R_root.T @ R_ee_world
    """
    with open(path) as f:
        data = json.load(f)

    frames = data["frames"]
    T = len(frames)
    if T < 2:
        return None

    out = np.zeros((T, cfg.raw_dim), dtype=np.float32)

    T_previous_root = None

    for i, fr in enumerate(frames):
        col = 0

        root_world_pos = np.array(fr["torso"]["pos"], dtype=np.float32)
        rq = fr["torso"]["rot"]
        q_root = np.array([rq["w"], rq["x"], rq["y"], rq["z"]], dtype=np.float32)

        T_root = t3d.affines.compose(
            root_world_pos,
            t3d.quaternions.quat2mat(q_root),
            np.ones(3))

        if T_previous_root is not None:
            T_delta = np.linalg.inv(T_previous_root) @ T_root
        else:
            T_delta = np.eye(4)
        T_previous_root = T_root.copy()

        out[i, col:col + 1] = T_root[2, 3]  # root Z height in world space (not delta)
        col += 1

        out[i, col:col + 3] = T_delta[:3, 3]   # root pos delta
        col += 3

        out[i, col:col + 4] = t3d.quaternions.mat2quat(T_delta[:3, :3]) # root rot delta
        col += 4


        # end effectors in root-local frame
        for ee_key in EE_KEYS:
            ee           = fr[ee_key]
            ee_world_pos = np.array(ee["pos"], dtype=np.float32)
            eq           = ee["rot"]
            q_ee_world   = np.array([eq["w"], eq["x"], eq["y"], eq["z"]], dtype=np.float32)

            T_ee = t3d.affines.compose(
                ee_world_pos,
                t3d.quaternions.quat2mat(q_ee_world),
                np.ones(3))

            T_local_ee = np.linalg.inv(T_root) @ T_ee

            out[i, col:col + 3] = T_local_ee[:3, 3]
            col += 3
            out[i, col:col + 4] = t3d.quaternions.mat2quat(T_local_ee[:3, :3])
            col += 4

    return out   # (T, 36)


def sliding_windows(clip: np.ndarray, window: int) -> list[np.ndarray]:
    T = len(clip)
    if T < window:
        return []
    return [clip[s : s + window] for s in range(T - window + 1)]


class MotionDataset(Dataset):
    """
    Stride-1 sliding-window dataset over pre-split raw (quaternion) clips.

    Per __getitem__:
      1. Copy window (never mutate stored views).
      2. Convert quaternions -> 6D rotations.
      3. Normalize to [-1, 1] using training-set statistics.
    """

    def __init__(
        self,
        clips: list[np.ndarray],
        max_seq_len: int,
        feat_min: np.ndarray,
        feat_max: np.ndarray,
        feat_range: np.ndarray,
    ):
        self.max_seq_len = max_seq_len
        self.feat_min    = feat_min
        self.feat_max    = feat_max
        self.feat_range  = feat_range
        self.windows: list[np.ndarray] = []
        print("Loading dataset into ram...")
        for clip in tqdm(clips):
            for w in sliding_windows(clip, max_seq_len):
                feat = raw_to_feat(w)
                feat = self.normalize(feat)
                self.windows.append(torch.from_numpy(feat))

        self.cache = {}

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.feat_min) / self.feat_range - 1.0

    def denormalize(self, x):
        mn  = torch.tensor(self.feat_min,   dtype=torch.float32)
        rng = torch.tensor(self.feat_range, dtype=torch.float32)
        if isinstance(x, torch.Tensor):
            return (x + 1.0) / 2.0 * rng + mn
        return (x + 1.0) / 2.0 * self.feat_range + self.feat_min

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def load_all_clips(data_dir: str, min_seq_len: int) -> list[np.ndarray]:
    print(f"Loading clips from {data_dir}...")
    clips = []
    for p in tqdm(sorted(Path(data_dir).glob("*.json"))):
        clip = load_json_clip(p)
        if clip is not None and len(clip) >= min_seq_len:
            clips.append(clip)
    return clips


def clip_level_split(clips, val_split, seed):
    """
    Shuffle and split at clip level so no windows from the same clip
    appear in both train and val (despite stride-1 overlap within clips).
    """
    rng     = random.Random(seed)
    indices = list(range(len(clips)))
    rng.shuffle(indices)
    n_val   = max(1, int(len(clips) * val_split))
    val_idx = set(indices[:n_val])
    return (
        [c for i, c in enumerate(clips) if i not in val_idx],
        [c for i, c in enumerate(clips) if i in val_idx],
    )


def compute_norm_stats_process_clip_windows(clip: np.ndarray) -> np.ndarray:
    """Process all windows from a single clip. Module-level for pickling."""
    frames = []
    for w in sliding_windows(clip, cfg.max_seq_len):
        frames.append(raw_to_feat(w))
    return frames

def compute_norm_stats(train_clips):
    """
    Compute per-feature min/max/range over all canonicalized, 6D-converted
    training windows. Stats are in feat-space (46-dim), matching model input.
    """
    print("Computing normalisation statistics over training windows...")

    n_workers = min(len(train_clips), os.cpu_count() or 1)
    with Pool(processes=n_workers) as pool:
        results = pool.map(compute_norm_stats_process_clip_windows, train_clips)

    feat_frames = []
    for clip_frames in results:
        feat_frames.extend(clip_frames)

    all_feats  = np.concatenate(feat_frames, axis=0)
    feat_min   = all_feats.min(axis=0)
    feat_max   = all_feats.max(axis=0)

    feat_range = np.where(feat_max - feat_min > 1e-8, feat_max - feat_min, 1.0)

    return feat_min, feat_max, feat_range

# ─────────────────────────────────────────
#  DIFFUSION SCHEDULE
# ─────────────────────────────────────────

class GaussianDiffusion:
    def __init__(self, n_timesteps: int, beta_start: float, beta_end: float, device: str):
        self.T      = n_timesteps
        self.device = device

        betas              = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)

        self.register("betas",                        betas)
        self.register("alphas_cumprod",               alphas_cumprod)
        self.register("sqrt_alphas_cumprod",          alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod",(1.0 - alphas_cumprod).sqrt())

    def register(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor.to(self.device))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha     = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def p_losses(self, model: nn.Module, x0: torch.Tensor,
                 loss_type: str = "l2") -> torch.Tensor:
        B  = x0.shape[0]
        t  = torch.randint(0, self.T, (B,), device=self.device)
        x0_perturbed = x0 + cfg.input_perturb_std * torch.randn_like(x0)
        xt, noise    = self.q_sample(x0_perturbed, t)
        pred_noise   = model(xt, t)

        if loss_type == "l2":
            return F.mse_loss(pred_noise, noise)
        elif loss_type == "l1":
            return F.l1_loss(pred_noise, noise)
        raise ValueError(f"Unknown loss type: {loss_type}")

# ─────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class TrajectoryDiffusionTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.feat_dim    = cfg.feat_dim
        self.d_model     = cfg.d_model
        self.max_seq_len = cfg.max_seq_len

        self.time_emb   = SinusoidalTimestepEmbedding(cfg.d_model)
        self.input_proj = nn.Linear(cfg.feat_dim, cfg.d_model)
        self.pos_emb    = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.nhead,
            dim_feedforward = cfg.dim_feedforward,
            dropout         = cfg.dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.feat_dim),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = xt.shape
        x       = self.input_proj(xt)
        x       = x + self.pos_emb(torch.arange(T, device=xt.device))[None]
        x       = x + self.time_emb(t)[:, None, :]
        x       = self.transformer(x)
        return self.output_proj(x)

# ─────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────

def train():
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # ── Load all clips, split at clip level ───────────────────────────
    all_clips = load_all_clips(cfg.data_dir, cfg.min_seq_len)
    if not all_clips:
        raise RuntimeError(f"No valid clips found in {cfg.data_dir}")

    train_clips, val_clips = clip_level_split(all_clips, cfg.val_split, cfg.seed)
    print(f"Clips — train: {len(train_clips)} | val: {len(val_clips)}")

    # ── Normalisation stats from training data only ───────────────────
    feat_min, feat_max, feat_range = compute_norm_stats(train_clips)

    print(f"{feat_min=} {feat_max=} {feat_range=}")

    # ── Build datasets ────────────────────────────────────────────────
    train_set = MotionDataset(train_clips, cfg.max_seq_len, feat_min, feat_max, feat_range)
    val_set   = MotionDataset(val_clips,   cfg.max_seq_len, feat_min, feat_max, feat_range)

    print(f"Windows — train: {len(train_set)} | val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # ── Model, EMA, diffusion ─────────────────────────────────────────
    model     = TrajectoryDiffusionTransformer(cfg).to(device)
    ema       = EMA(model, decay=cfg.ema_decay)
    diffusion = GaussianDiffusion(cfg.n_timesteps, cfg.beta_start, cfg.beta_end, cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss_acc = 0.0

        for batch in tqdm(train_loader, leave=False):
            batch = batch.to(device, non_blocking=True)
            loss  = diffusion.p_losses(model, batch, cfg.loss_type)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)

            train_loss_acc += loss.detach().item()

        scheduler.step()
        train_loss = train_loss_acc / len(train_loader)

        # ── Validation (EMA weights) ───────────────────────────────────
        model.eval()
        val_loss_acc = 0.0
        with ema.average_parameters(model):
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    loss  = diffusion.p_losses(model, batch, cfg.loss_type)
                    val_loss_acc += loss.item()
        val_loss = val_loss_acc / len(val_loader)

        if epoch % cfg.log_every == 0:
            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"train {train_loss:.5f} | val(EMA) {val_loss:.5f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )

        # ── Checkpointing (always save EMA weights) ───────────────────
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % cfg.save_every == 0 or is_best:
            state = {
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "ema_state":        ema.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "val_loss":         val_loss,
                "feat_min":         feat_min,
                "feat_max":         feat_max,
                "feat_range":       feat_range,
                "cfg":              cfg.__dict__,
            }
            if is_best:
                torch.save(state, ckpt_dir / "best.pt")
            if epoch % cfg.save_every == 0:
                torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")

    print(f"Training complete. Best val loss (EMA): {best_val_loss:.5f}")


# ─────────────────────────────────────────
#  SAMPLING  (DDPM reverse process)
# ─────────────────────────────────────────

@torch.no_grad()
def sample(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    n_samples: int,
    seq_len: int,
    feat_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Full DDPM reverse pass. Returns (n_samples, seq_len, feat_dim) normalized."""
    model.eval()
    x = torch.randn(n_samples, seq_len, feat_dim, device=device)

    for t_idx in reversed(range(diffusion.T)):
        t          = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        beta           = diffusion.betas[t_idx]
        alpha_bar      = diffusion.alphas_cumprod[t_idx]
        alpha_bar_prev = diffusion.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.ones(1, device=device)

        sqrt_alpha_bar = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).sqrt()
        x0_pred        = ((x - sqrt_one_minus * pred_noise) / sqrt_alpha_bar).clamp(-1.0, 1.0)

        coef1 = beta * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)
        coef2 = (1.0 - alpha_bar_prev) * (1.0 - beta).sqrt() / (1.0 - alpha_bar)
        mean  = coef1 * x0_pred + coef2 * x

        if t_idx > 0:
            posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            x = mean + posterior_var.sqrt() * torch.randn_like(x)
        else:
            x = mean

    return x

# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    train()
