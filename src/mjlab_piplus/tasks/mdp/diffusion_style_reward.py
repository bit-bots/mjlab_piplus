"""
diffusion_style_reward.py
=========================
mjlab reward term using a pre-trained motion-diffusion model as a style critic.

Scale & frame-convention bridge
--------------------------------
The diffusion model was trained on motion-capture data for a full-size human
(~1 m waist height, SI units).  The MuJoCo robot may be a scaled-down version
(e.g. SCALE = 0.25 → ~25 cm waist height).  To feed the model correctly:

  • All translations extracted from MuJoCo are multiplied by (1 / SCALE)
    before being stored in the feature buffer, so they match the model's
    training distribution.
  • The reward error is computed entirely in model space (large-robot space),
    so no inverse scaling of the error is needed.

The model was also trained with a specific EE frame convention that differs
from the raw MuJoCo site frames.  For each end-effector a fixed rotation
T_fix is applied (matching the demo.py decode_trajectory logic), and for
feet/hands a lateral stance shift is added in root-local space.  Both are
applied after extracting raw MuJoCo poses and before building the feature
vector, so the buffer always contains model-convention features.

Specifically (mirroring demo.py):
  side = +1 for right EEs, -1 for left EEs
  T_fix rotation:
    feet  → Rx(-side · 20°)
    hands → Rx(-side · 90°)
  Stance shift (root-local Y):
    feet/hands → +STANCE_SHIFT * side   (right = +Y, left = -Y)
    (note: demo shifts in the EE-world frame but since we work root-local
     the equivalent is a root-local Y offset of ±STANCE_SHIFT)

MuJoCo site names
------------------
The EE sites in the MJCF are named exactly:  l_hand, r_hand, l_foot, r_foot.
No user-supplied mapping is required; the term resolves them automatically.

Buffer strategy
---------------
Each environment has a write-pointer and a fixed buffer of max_seq_len frames.

  Phase 1 – Fill:   write one feature frame per step until ptr == max_seq_len.
  Phase 2 – Score:  run diffusion critic on full buffer → emit reward (once).
  Phase 3 – Halve:  keep second half as context seed, reset ptr to half_len.
  Repeat.

Reward is 0 on every step that is not a scoring step.
After the initial warm-up scoring fires every (max_seq_len // 2) steps.

Scoring (single-step denoiser-as-critic)
-----------------------------------------
For each noise level α ∈ noise_levels (e.g. 0.1, 0.3, 0.7):
  x_noisy = √ᾱ · x_norm  +  √(1-ᾱ) · ε,   ε ~ N(0, I)
  x̂₀      = denoiser(x_noisy, t)             ← ONE forward pass
  Δ        = x̂₀ − x_norm

Δ is split into six independently-scaled components:

  root_vel_trans  feat[:,1:4]        Euclidean distance (in model-space metres)
  root_vel_rot    feat[:,4:10] 6D    Min quaternion angle / π
  foot_trans      l_foot/r_foot pos  Euclidean distance (averaged)
  foot_rot        l_foot/r_foot 6D   Min quaternion angle / π (averaged)
  hand_trans      l_hand/r_hand pos  Euclidean distance (averaged)
  hand_rot        l_hand/r_hand 6D   Min quaternion angle / π (averaged)

  combined = Σ_level  weight · (
      s_root_vt · err_root_trans
    + s_root_vr · err_root_rot/π
    + s_foot_t  · err_foot_trans
    + s_foot_r  · err_foot_rot/π
    + s_hand_t  · err_hand_trans
    + s_hand_r  · err_hand_rot/π
  )
  reward = exp(−reward_scale · combined)

Usage
-----
    from diffusion_style_reward import DiffusionStyleReward

    @configclass
    class RewardsCfg:
        diffusion_style = RewardTermCfg(
            func=DiffusionStyleReward,
            weight=1.0,
            params=dict(
                checkpoint_path="/path/to/best.pt",
                root_body_name="pelvis",
                # SCALE: robot size relative to training data (e.g. 0.25 = quarter size)
                robot_scale=0.25,
                # lateral stance offset added in root-local Y (model-space metres)
                stance_shift=0.2,
                noise_levels=(0.1, 0.3, 0.7),
                level_weights=(0.5, 0.3, 0.2),
                scale_root_vel_trans=1.0,
                scale_root_vel_rot=1.0,
                scale_foot_trans=1.0,
                scale_foot_rot=1.0,
                scale_hand_trans=1.0,
                scale_hand_rot=1.0,
                reward_scale=5.0,
            ),
        )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

# ── mjlab imports ──────────────────────────────────────────────────────────────
from mjlab.managers import ManagerTermBase, RewardTermCfg  # type: ignore

# ── Diffusion model imports (train.py must be importable) ─────────────────────
from mjlab_piplus.tasks.mdp.diffusion_train_script import (  # type: ignore
    Config as DiffusionConfig,
    GaussianDiffusion,
    TrajectoryDiffusionTransformer,
    EE_KEYS,
    rot6d_to_rotmat_torch,
)

# EE_KEYS must be ["l_foot", "r_foot", "l_hand", "r_hand"] — order matters for
# the feature layout.  The MuJoCo site names match these keys exactly.
_FOOT_KEYS: Tuple[str, ...] = ("l_foot", "r_foot")
_HAND_KEYS: Tuple[str, ...] = ("l_hand", "r_hand")

# Per-EE fix rotation angles (degrees) mirroring demo.py T_fix logic.
# For each key: Rx(−side · angle_deg) where side = +1 right, −1 left.
_EE_FIX_ANGLE_DEG: Dict[str, float] = {
    "l_foot": 20.0,
    "r_foot": 20.0,
    "l_hand": 90.0,
    "r_hand": 90.0,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wxyz_quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz → (N, 3, 3) rotation matrix."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """
    (*, 3, 3) → (*, 4) unit quaternion (w, x, y, z).
    Shepperd method, fully vectorised.
    """
    orig = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    def _ss(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(t, min=1e-10))

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    s1 = _ss(tr + 1.0) * 2.0
    s2 = _ss(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) * 2.0
    s3 = _ss(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]) * 2.0
    s4 = _ss(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]) * 2.0

    q1 = torch.stack([0.25*s1,
                      (R[:,2,1]-R[:,1,2])/s1,
                      (R[:,0,2]-R[:,2,0])/s1,
                      (R[:,1,0]-R[:,0,1])/s1], dim=-1)
    q2 = torch.stack([(R[:,2,1]-R[:,1,2])/s2,
                      0.25*s2,
                      (R[:,0,1]+R[:,1,0])/s2,
                      (R[:,0,2]+R[:,2,0])/s2], dim=-1)
    q3 = torch.stack([(R[:,0,2]-R[:,2,0])/s3,
                      (R[:,0,1]+R[:,1,0])/s3,
                      0.25*s3,
                      (R[:,1,2]+R[:,2,1])/s3], dim=-1)
    q4 = torch.stack([(R[:,1,0]-R[:,0,1])/s4,
                      (R[:,0,2]+R[:,2,0])/s4,
                      (R[:,1,2]+R[:,2,1])/s4,
                      0.25*s4], dim=-1)

    c1 = (tr > 0).unsqueeze(-1)
    c2 = ((R[:,0,0] > R[:,1,1]) & (R[:,0,0] > R[:,2,2]) & ~c1.squeeze(-1)).unsqueeze(-1)
    c3 = ((R[:,1,1] > R[:,2,2]) & ~c1.squeeze(-1) & ~c2.squeeze(-1)).unsqueeze(-1)

    q = torch.where(c1, q1, torch.where(c2, q2, torch.where(c3, q3, q4)))
    return F.normalize(q, dim=-1).reshape(*orig, 4)


def _rot6d_to_quat(r6d: torch.Tensor) -> torch.Tensor:
    """(*, 6) → (*, 4) unit quaternion (w, x, y, z)."""
    R = rot6d_to_rotmat_torch(r6d.reshape(-1, 6))
    return _rotmat_to_quat_wxyz(R).reshape(*r6d.shape[:-1], 4)


def _quat_angle_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Minimum rotation angle between unit quaternions (w, x, y, z).
    angle = 2 · arccos(|q1·q2|)  in radians,  shape (*,).
    """
    dot = torch.clamp(torch.abs((q1 * q2).sum(dim=-1)), 0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _rx_matrix(angle_rad: float, device: torch.device) -> torch.Tensor:
    """3×3 rotation matrix around the X axis."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0,   c,  -s],
         [0.0,   s,   c]],
        dtype=torch.float32, device=device,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Pre-compute per-EE fix rotations (done once at module load time per device)
# ──────────────────────────────────────────────────────────────────────────────

def _build_ee_fix_rotations(device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Return {ee_key: R_fix (3,3)} matching demo.py T_fix.

    demo.py applies:
        T_fix[:3,:3] = euler2mat(radians(-side * angle_deg), 0, 0)
    which is Rx(-side * angle_deg).

    side = +1 for right (r_*), -1 for left (l_*).
    """
    fixes: Dict[str, torch.Tensor] = {}
    for key, angle_deg in _EE_FIX_ANGLE_DEG.items():
        side = 1.0 if key.startswith("r_") else -1.0
        angle_rad = math.radians(-side * angle_deg)
        fixes[key] = _rx_matrix(angle_rad, device)
    return fixes


# ──────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_policy_features(
    env,
    root_body_id: int,
    ee_site_ids: Dict[str, int],
    prev_root_pos:   torch.Tensor,          # (N, 3)  MuJoCo world-space (unscaled)
    prev_root_quat:  torch.Tensor,          # (N, 4)  wxyz, MuJoCo world-space
    inv_scale: float,                       # 1.0 / robot_scale  (e.g. 4.0)
    stance_shift: float,                    # lateral offset in model-space metres
    ee_fix_rots: Dict[str, torch.Tensor],   # pre-built per-EE fix rotations
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the 46-dim feature vector (in model/training space) from sim state.

    All translations are multiplied by inv_scale so the model sees full-size
    human-scale numbers regardless of the robot's actual size.
    EE orientations have the per-EE frame-convention fix applied.
    EE positions have the lateral stance shift applied in root-local space.

    Returns
    -------
    feat            : (N_env, 46)  in model space, ready for normalisation
    new_root_pos    : (N_env, 3)   raw MuJoCo world-space pos (cache for next step)
    new_root_quat   : (N_env, 4)   raw MuJoCo world-space quat wxyz (cache)
    """
    N = env.num_envs

    # Raw MuJoCo poses (unscaled world-space)
    xpos  = env.sim.data.xpos    # (N, nbody, 3)
    xquat = env.sim.data.xquat   # (N, nbody, 4) wxyz

    root_pos_raw  = xpos[:,  root_body_id, :]   # (N, 3) — keep unscaled for caching
    root_quat_raw = xquat[:, root_body_id, :]   # (N, 4) wxyz

    # Rotation matrix of the root (same in both spaces — rotations don't scale)
    R_cur  = _wxyz_quat_to_rotmat(root_quat_raw)    # (N, 3, 3)
    R_prev = _wxyz_quat_to_rotmat(prev_root_quat)   # (N, 3, 3)

    # ── feat[0]: absolute Z height, scaled to model space ─────────────────────
    z_height_model = root_pos_raw[:, 2:3] * inv_scale              # (N, 1)

    # ── feat[1:4]: root Δpos in prev-root-local frame, scaled ─────────────────
    # Δpos_world (raw) → express in prev-root-local frame → scale up
    delta_pos_world_raw = root_pos_raw - prev_root_pos              # (N, 3) unscaled
    delta_pos_local_model = torch.bmm(
        R_prev.transpose(1, 2),
        delta_pos_world_raw.unsqueeze(-1),
    ).squeeze(-1) * inv_scale                                       # (N, 3)

    # ── feat[4:10]: root Δrot 6D (rotation is scale-invariant) ───────────────
    R_delta = torch.bmm(R_prev.transpose(1, 2), R_cur)             # (N, 3, 3)
    rot6d_root = torch.cat(
        [R_delta[:, :, 0], R_delta[:, :, 1]], dim=-1)               # (N, 6)

    # ── feat[10:46]: EE features ──────────────────────────────────────────────
    ee_feats: List[torch.Tensor] = []
    for key in EE_KEYS:
        ee_id         = ee_site_ids[key]
        ee_pos_raw    = xpos[:,  ee_id, :]                          # (N, 3) unscaled
        ee_quat_world = xquat[:, ee_id, :]                         # (N, 4) wxyz

        # ── EE position: root-local, scaled, + lateral stance shift ───────────
        # 1. Vector from root to EE in world space (raw)
        # 2. Rotate into current root-local frame
        # 3. Scale to model space
        ee_pos_local_model = torch.bmm(
            R_cur.transpose(1, 2),
            (ee_pos_raw - root_pos_raw).unsqueeze(-1),
        ).squeeze(-1) * inv_scale                                   # (N, 3)

        # Apply stance shift: shift EE along root-local Y axis.
        # Right EEs shift +Y, left EEs shift −Y (matches demo.py stance logic).
        side = 1.0 if key.startswith("r_") else -1.0
        ee_pos_local_model = ee_pos_local_model.clone()
        ee_pos_local_model[:, 1] = ee_pos_local_model[:, 1] + side * stance_shift

        # ── EE orientation: root-local + frame-convention fix ─────────────────
        # R_ee_local_raw = R_cur^T @ R_ee_world
        # Then apply the per-EE fix:  R_ee_local_fixed = R_ee_local_raw @ R_fix
        # This mirrors:  T_eef_world = T_root @ T_eef @ T_fix  in demo.py,
        # which in rotation-only terms gives:
        #   R_eef_world_fixed = R_root @ R_eef_local_raw @ R_fix
        # So in root-local:
        #   R_eef_local_fixed = R_eef_local_raw @ R_fix
        R_ee_world = _wxyz_quat_to_rotmat(ee_quat_world)           # (N, 3, 3)
        R_ee_local_raw = torch.bmm(
            R_cur.transpose(1, 2), R_ee_world)                     # (N, 3, 3)

        R_fix = ee_fix_rots[key]                                    # (3, 3)
        # Broadcast R_fix across the batch
        R_ee_local_fixed = torch.bmm(
            R_ee_local_raw,
            R_fix.unsqueeze(0).expand(N, -1, -1),
        )                                                           # (N, 3, 3)

        rot6d_ee = torch.cat(
            [R_ee_local_fixed[:, :, 0],
             R_ee_local_fixed[:, :, 1]], dim=-1)                   # (N, 6)

        ee_feats.extend([ee_pos_local_model, rot6d_ee])

    feat = torch.cat(
        [z_height_model, delta_pos_local_model, rot6d_root, *ee_feats],
        dim=-1,
    )   # (N, 46)

    return feat, root_pos_raw.detach().clone(), root_quat_raw.detach().clone()


# ──────────────────────────────────────────────────────────────────────────────
#  Diffusion scoring  (single forward pass per noise level, no full denoising)
# ──────────────────────────────────────────────────────────────────────────────

def _score_trajectory_chunk(
    feat_chunk: torch.Tensor,            # (N, T, 46)  normalised to [-1, 1]
    model: TrajectoryDiffusionTransformer,
    diffusion: GaussianDiffusion,
    noise_levels: Sequence[float],
    level_weights: Sequence[float],
    scale_root_vel_trans: float,
    scale_root_vel_rot:   float,
    scale_foot_trans:     float,
    scale_foot_rot:       float,
    scale_hand_trans:     float,
    scale_hand_rot:       float,
    device: torch.device,
) -> torch.Tensor:
    """
    Weighted denoiser-as-critic error per environment, shape (N,).
    Lower = trajectory more in-distribution = better style.
    """
    N, T, _ = feat_chunk.shape
    total_error = torch.zeros(N, device=device)

    ee_col     = {key: 10 + k * 9 for k, key in enumerate(EE_KEYS)}
    foot_keys  = [k for k in EE_KEYS if k in _FOOT_KEYS]
    hand_keys  = [k for k in EE_KEYS if k in _HAND_KEYS]

    with torch.no_grad():
        for alpha, lw in zip(noise_levels, level_weights):
            # Nearest DDPM timestep whose ᾱₜ ≈ alpha
            t_idx = int(
                torch.argmin(torch.abs(diffusion.alphas_cumprod - alpha)).item()
            )
            t_tensor = torch.full((N,), t_idx, dtype=torch.long, device=device)

            # Forward noise (single step — no iteration)
            eps     = torch.randn_like(feat_chunk)
            x_noisy = math.sqrt(alpha) * feat_chunk + math.sqrt(1.0 - alpha) * eps

            # Single denoiser pass → predicted clean trajectory
            x_hat = model(x_noisy, t_tensor)    # (N, T, 46)
            delta = x_hat - feat_chunk           # (N, T, 46)

            # ── root velocity translation (feat[:,1:4]) ────────────────────────
            err_root_trans = delta[:, :, 1:4].norm(dim=-1).mean(dim=-1)    # (N,)

            # ── root velocity rotation (feat[:,4:10] as 6D) ───────────────────
            q_o_root = _rot6d_to_quat(feat_chunk[:, :, 4:10].reshape(N*T, 6)).reshape(N, T, 4)
            q_h_root = _rot6d_to_quat(x_hat[:,     :, 4:10].reshape(N*T, 6)).reshape(N, T, 4)
            err_root_rot = _quat_angle_diff(
                q_o_root.reshape(N*T, 4), q_h_root.reshape(N*T, 4)
            ).reshape(N, T).mean(dim=-1)                                    # (N,)

            # ── foot errors ───────────────────────────────────────────────────
            err_foot_trans = torch.zeros(N, device=device)
            err_foot_rot   = torch.zeros(N, device=device)
            for key in foot_keys:
                col = ee_col[key]
                err_foot_trans += delta[:, :, col:col+3].norm(dim=-1).mean(dim=-1)
                q_o = _rot6d_to_quat(feat_chunk[:, :, col+3:col+9].reshape(N*T, 6)).reshape(N, T, 4)
                q_h = _rot6d_to_quat(x_hat[:,     :, col+3:col+9].reshape(N*T, 6)).reshape(N, T, 4)
                err_foot_rot += _quat_angle_diff(
                    q_o.reshape(N*T, 4), q_h.reshape(N*T, 4)
                ).reshape(N, T).mean(-1)
            if foot_keys:
                err_foot_trans /= len(foot_keys)
                err_foot_rot   /= len(foot_keys)

            # ── hand errors ───────────────────────────────────────────────────
            err_hand_trans = torch.zeros(N, device=device)
            err_hand_rot   = torch.zeros(N, device=device)
            for key in hand_keys:
                col = ee_col[key]
                err_hand_trans += delta[:, :, col:col+3].norm(dim=-1).mean(dim=-1)
                q_o = _rot6d_to_quat(feat_chunk[:, :, col+3:col+9].reshape(N*T, 6)).reshape(N, T, 4)
                q_h = _rot6d_to_quat(x_hat[:,     :, col+3:col+9].reshape(N*T, 6)).reshape(N, T, 4)
                err_hand_rot += _quat_angle_diff(
                    q_o.reshape(N*T, 4), q_h.reshape(N*T, 4)
                ).reshape(N, T).mean(-1)
            if hand_keys:
                err_hand_trans /= len(hand_keys)
                err_hand_rot   /= len(hand_keys)

            # ── weighted combination ──────────────────────────────────────────
            # Rotation errors ∈ [0, π]; divide by π → [0, 1] to match
            # magnitude of translation errors (in model-space metres).
            combined = (
                scale_root_vel_trans * err_root_trans
                + scale_root_vel_rot  * err_root_rot  / math.pi
                + scale_foot_trans    * err_foot_trans
                + scale_foot_rot      * err_foot_rot  / math.pi
                + scale_hand_trans    * err_hand_trans
                + scale_hand_rot      * err_hand_rot  / math.pi
            )
            total_error = total_error + lw * combined

    return total_error   # (N,)


# ──────────────────────────────────────────────────────────────────────────────
#  Main reward term
# ──────────────────────────────────────────────────────────────────────────────

class DiffusionStyleReward(ManagerTermBase):
    """
    mjlab reward term: motion-diffusion style critic.

    See module docstring for full description of the scale bridge, frame-
    convention fix, and fill/score/halve buffer strategy.
    """

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(env)
        p = cfg.params

        # ── scale & frame convention ──────────────────────────────────────────
        self._inv_scale:    float = 1.0 / float(p.get("robot_scale",  0.25))
        self._stance_shift: float = float(p.get("stance_shift", 0.2))

        # Pre-build fix rotations for this device
        self._ee_fix_rots: Dict[str, torch.Tensor] = _build_ee_fix_rotations(self.device)

        # ── noise schedule ────────────────────────────────────────────────────
        self._noise_levels:  Tuple[float, ...] = tuple(p.get("noise_levels",  (0.1, 0.3, 0.7)))
        self._level_weights: Tuple[float, ...] = tuple(p.get("level_weights", (0.5, 0.3, 0.2)))
        assert len(self._noise_levels) == len(self._level_weights)
        assert abs(sum(self._level_weights) - 1.0) < 1e-5, "level_weights must sum to 1"

        # ── per-component error scales ────────────────────────────────────────
        self._scale_root_vel_trans: float = float(p.get("scale_root_vel_trans", 1.0))
        self._scale_root_vel_rot:   float = float(p.get("scale_root_vel_rot",   1.0))
        self._scale_foot_trans:     float = float(p.get("scale_foot_trans",     1.0))
        self._scale_foot_rot:       float = float(p.get("scale_foot_rot",       1.0))
        self._scale_hand_trans:     float = float(p.get("scale_hand_trans",     1.0))
        self._scale_hand_rot:       float = float(p.get("scale_hand_rot",       1.0))

        # ── reward shaping ────────────────────────────────────────────────────
        self._reward_scale: float = float(p.get("reward_scale", 5.0))

        # ── body / site ID resolution ─────────────────────────────────────────
        # Root is a body; EEs are sites named l_foot, r_foot, l_hand, r_hand.
        body_names = list(env.sim.model.body_names)
        site_names = list(env.sim.model.site_names)

        def _body_id(name: str) -> int:
            if name not in body_names:
                raise ValueError(
                    f"DiffusionStyleReward: body '{name}' not found. "
                    f"Available: {body_names}"
                )
            return body_names.index(name)

        def _site_id(name: str) -> int:
            if name not in site_names:
                raise ValueError(
                    f"DiffusionStyleReward: site '{name}' not found. "
                    f"Available: {site_names}"
                )
            return site_names.index(name)

        self._root_body_id: int = _body_id(p["root_body_name"])

        # EE sites: MuJoCo site names == EE_KEYS (l_foot, r_foot, l_hand, r_hand)
        self._ee_site_ids: Dict[str, int] = {key: _site_id(key) for key in EE_KEYS}

        # ── load diffusion checkpoint ─────────────────────────────────────────
        ckpt_path = Path(p["checkpoint_path"])
        print(f"[DiffusionStyleReward] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        dcfg = DiffusionConfig()
        for k, v in ckpt["cfg"].items():
            if hasattr(dcfg, k):
                setattr(dcfg, k, v)

        self._max_seq_len: int = dcfg.max_seq_len
        self._half_len:    int = dcfg.max_seq_len // 2
        self._feat_dim:    int = dcfg.feat_dim       # 46

        self._feat_min:   torch.Tensor = torch.tensor(
            ckpt["feat_min"],   dtype=torch.float32, device=self.device)
        self._feat_range: torch.Tensor = torch.tensor(
            ckpt["feat_range"], dtype=torch.float32, device=self.device)

        model = TrajectoryDiffusionTransformer(dcfg).to(self.device)
        state = ckpt.get("ema_state", ckpt.get("model_state"))
        model.load_state_dict(state)
        model.eval()
        self._model = model

        self._diffusion = GaussianDiffusion(
            dcfg.n_timesteps, dcfg.beta_start, dcfg.beta_end, str(self.device)
        )

        print(
            f"[DiffusionStyleReward] Ready.  "
            f"epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.5f}  "
            f"max_seq_len={self._max_seq_len}  inv_scale={self._inv_scale:.2f}  "
            f"stance_shift={self._stance_shift}  noise_levels={self._noise_levels}  "
            f"scores every {self._half_len} steps after warm-up"
        )

        # ── per-environment buffer state ──────────────────────────────────────
        N_envs = env.num_envs

        # Feature buffer in model space: (N_envs, max_seq_len, feat_dim)
        self._buf: torch.Tensor = torch.zeros(
            N_envs, self._max_seq_len, self._feat_dim,
            dtype=torch.float32, device=self.device,
        )

        # Write pointer per env: next frame → buf[env_i, ptr[env_i], :]
        self._ptr: torch.Tensor = torch.zeros(
            N_envs, dtype=torch.long, device=self.device)

        # Previous root pose stored in raw MuJoCo world-space (unscaled),
        # because xpos is unscaled and we need raw Δpos to be correct before
        # applying inv_scale inside _extract_policy_features.
        self._prev_root_pos:  torch.Tensor = torch.zeros(
            N_envs, 3, dtype=torch.float32, device=self.device)
        self._prev_root_quat: torch.Tensor = torch.zeros(
            N_envs, 4, dtype=torch.float32, device=self.device)
        self._prev_root_quat[:, 0] = 1.0   # identity quaternion (w=1)

    # ── mjlab hooks ───────────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor) -> None:
        """Called by mjlab at episode boundaries for the given env indices."""
        self._buf[env_ids] = 0.0
        self._ptr[env_ids] = 0

        # Seed previous root pose from the freshly-reset sim state
        xpos  = self.env.sim.data.xpos
        xquat = self.env.sim.data.xquat
        self._prev_root_pos[env_ids]  = xpos[ env_ids, self._root_body_id, :].detach().clone()
        self._prev_root_quat[env_ids] = xquat[env_ids, self._root_body_id, :].detach().clone()

    def __call__(self, env, **kwargs) -> torch.Tensor:
        """
        Called every simulation step by mjlab's RewardManager.

        Returns (N_envs,) float tensor.
        Non-zero only on the step a full buffer is scored (every half_len steps
        after the initial warm-up).
        """
        N = env.num_envs

        # ── 1. Extract model-space features ───────────────────────────────────
        feat, self._prev_root_pos, self._prev_root_quat = _extract_policy_features(
            env,
            self._root_body_id,
            self._ee_site_ids,
            self._prev_root_pos,
            self._prev_root_quat,
            inv_scale=self._inv_scale,
            stance_shift=self._stance_shift,
            ee_fix_rots=self._ee_fix_rots,
        )   # (N, 46)  in model space

        # ── 2. Scatter-write into buffer (vectorised, no loop) ─────────────────
        env_idx = torch.arange(N, device=self.device)
        self._buf[env_idx, self._ptr] = feat
        self._ptr += 1

        # ── 3. Default reward is zero ──────────────────────────────────────────
        reward = torch.zeros(N, device=self.device)

        # ── 4. Score envs whose buffer just filled ─────────────────────────────
        full_mask = (self._ptr >= self._max_seq_len)
        if full_mask.any():
            full_idx = full_mask.nonzero(as_tuple=True)[0]   # (V,)

            # Normalise to [-1, 1] for the diffusion model
            traj_raw  = self._buf[full_idx]                   # (V, T, D)
            traj_norm = (
                (traj_raw - self._feat_min)
                / self._feat_range.clamp(min=1e-8)
                * 2.0 - 1.0
            )

            error = _score_trajectory_chunk(
                traj_norm,
                self._model,
                self._diffusion,
                self._noise_levels,
                self._level_weights,
                scale_root_vel_trans=self._scale_root_vel_trans,
                scale_root_vel_rot=self._scale_root_vel_rot,
                scale_foot_trans=self._scale_foot_trans,
                scale_foot_rot=self._scale_foot_rot,
                scale_hand_trans=self._scale_hand_trans,
                scale_hand_rot=self._scale_hand_rot,
                device=self.device,
            )   # (V,)

            reward[full_idx] = torch.exp(-self._reward_scale * error)

            # ── 5. Keep second half as context seed, reset pointer ─────────────
            # buf[:, :half]  ← buf[:, half:T]   (recent context carried forward)
            # buf[:, half:]  ← 0
            # ptr            ← half_len
            self._buf[full_idx, :self._half_len, :] = (
                self._buf[full_idx, self._half_len:, :].clone()
            )
            self._buf[full_idx, self._half_len:, :] = 0.0
            self._ptr[full_idx] = self._half_len

        return reward
