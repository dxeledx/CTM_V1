"""
EEG-adapted CTM core (ticks + recursive synchronisation pairs + cross-attention).

We keep CTM's original code untouched and implement an EEG-facing "core" that matches
the design.md specs while reusing CTM's Neuron-Level Model building block (SuperLinear).

Design doc mapping:
  - design.md "CTM v1 默认规模"
  - design.md "Fusion: concat/film/gated"
  - design.md "Pair 采样策略" + "递推同步（α/β）"
  - design.md "归一化：LN 放哪里"
  - design.md "Tick-wise Head 设计" + "Certainty 定义"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_ctm.utils.ctm_repo import ensure_ctm_on_path


# Import CTM building blocks.
ensure_ctm_on_path()
from models.modules import SuperLinear  # noqa: E402


FusionMode = Literal["concat", "film", "gated"]
KVProjectorType = Literal["identity", "linear", "mlp"]
HeadType = Literal["linear", "2fc"]
InitMode = Literal["zeros", "learnable", "learnable_noise"]
TickLossType = Literal["mean_ce", "ctm_t1t2", "hybrid"]
ReadoutType = Literal["last", "most_certain", "mean_logits", "certainty_weighted"]


@dataclass(frozen=True)
class CTMCoreConfig:
    # Core dimensions
    D: int = 256
    T_internal: int = 12
    M_hist: int = 16

    # Attention dims
    d_input: int = 128
    n_heads: int = 8
    attn_dropout: float = 0.0

    # KV projector (Tokenizer d_kv -> CTM d_input)
    kv_projector: KVProjectorType = "identity"
    kv_mlp_hidden: int = 256

    # Synapse and NLM
    synapse_hidden: int = 512
    synapse_dropout: float = 0.1
    deep_nlms: bool = True
    memory_hidden_dims: int = 64
    do_layernorm_nlm: bool = False
    nlm_dropout: float = 0.0

    # Stabilization
    init_mode: InitMode = "zeros"
    ln_on_preact_hist: bool = False

    # Fusion
    fusion: FusionMode = "gated"

    # Synchronisation decay init
    no_decay_init: bool = True  # set raw_r ~ -10 so softplus(raw_r) ~ 0

    # Head
    head_type: HeadType = "linear"
    head_hidden: int = 128
    num_classes: int = 4


class CTMInit(nn.Module):
    """
    Design.md: "初始化：z_init 与 pre_acts_history_init（默认：零初始化）"
    """

    def __init__(self, D: int, M: int, mode: InitMode = "zeros") -> None:
        super().__init__()
        self.mode = mode
        if mode in ("learnable", "learnable_noise"):
            self.z_init = nn.Parameter(torch.zeros(D))
            self.A_init = nn.Parameter(torch.zeros(D, M))
        else:
            self.register_buffer("z_init", torch.zeros(D))
            self.register_buffer("A_init", torch.zeros(D, M))

    def make(self, B: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1).contiguous()
        A = self.A_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
        if self.mode == "learnable_noise":
            z = z + 0.01 * torch.randn_like(z)
            A = A + 0.01 * torch.randn_like(A)
        return z, A


class CTMNorm(nn.Module):
    """
    Design.md: "归一化：LN 放哪里（默认：LN on z + LN on o）"
    """

    def __init__(self, D: int, d_input: int, ln_on_preact_hist: bool = False) -> None:
        super().__init__()
        self.z_ln = nn.LayerNorm(D)
        self.o_ln = nn.LayerNorm(d_input)
        self.ln_on_preact_hist = ln_on_preact_hist
        self.A_ln = nn.LayerNorm(D) if ln_on_preact_hist else None

    def norm_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_ln(z)

    def norm_o(self, o: torch.Tensor) -> torch.Tensor:
        return self.o_ln(o)

    def norm_A(self, A: torch.Tensor) -> torch.Tensor:
        if not self.ln_on_preact_hist:
            return A
        assert self.A_ln is not None
        B, D, M = A.shape
        x = A.permute(0, 2, 1).reshape(B * M, D)
        x = self.A_ln(x)
        return x.reshape(B, M, D).permute(0, 2, 1)


class Fusion(nn.Module):
    """
    Design.md: "Fusion：concat/film/gated（默认 gated）"
    """

    def __init__(self, D: int, d_input: int, mode: FusionMode = "gated") -> None:
        super().__init__()
        self.mode = mode
        if mode == "film":
            self.film = nn.Sequential(
                nn.Linear(d_input, 2 * D),
                nn.GELU(),
                nn.Linear(2 * D, 2 * D),
            )
            self.z_ln = nn.LayerNorm(D)
        elif mode == "gated":
            self.gate = nn.Linear(D + d_input, D)
            self.cand = nn.Linear(D + d_input, D)

    def forward(self, z: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return torch.cat([z, o], dim=-1)
        if self.mode == "film":
            gb = self.film(o)
            gamma, beta = gb.chunk(2, dim=-1)
            z_mod = gamma * self.z_ln(z) + beta
            return torch.cat([z_mod, o], dim=-1)
        if self.mode == "gated":
            u = torch.cat([z, o], dim=-1)
            g = torch.sigmoid(self.gate(u))
            h = torch.tanh(self.cand(u))
            z_bar = (1.0 - g) * z + g * h
            return torch.cat([z_bar, o], dim=-1)
        raise ValueError(self.mode)


class RecursivePairSynchronizer(nn.Module):
    """
    Exponentially-decayed synchronisation over a subsampled set of neuron pairs.

    Implements design.md and CTM Appendix H recurrence:
      alpha_{t+1} = exp(-r) * alpha_t + z_{t+1,i} z_{t+1,j}
      beta_{t+1}  = exp(-r) * beta_t  + 1
      synch_t     = alpha_t / sqrt(beta_t)
    """

    def __init__(self, D: int, left_idx: torch.Tensor, right_idx: torch.Tensor, *, eps: float = 1e-8) -> None:
        super().__init__()
        if left_idx.shape != right_idx.shape:
            raise ValueError("left/right shape mismatch")
        self.D = int(D)
        self.Dsub = int(left_idx.numel())
        self.eps = float(eps)
        self.register_buffer("left", left_idx.long())
        self.register_buffer("right", right_idx.long())
        self.raw_r = nn.Parameter(torch.zeros(self.Dsub))

    def set_no_decay_init(self) -> None:
        with torch.no_grad():
            self.raw_r.fill_(-10.0)

    def init_state(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prod0 = z0[:, self.left] * z0[:, self.right]  # [B, Dsub]
        alpha = prod0
        beta = torch.ones_like(alpha)
        synch = alpha / torch.sqrt(beta + self.eps)
        return alpha, beta, synch

    def step(self, z_next: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = F.softplus(self.raw_r)  # [Dsub], r>=0
        decay = torch.exp(-r).unsqueeze(0)  # [1,Dsub]
        prod = z_next[:, self.left] * z_next[:, self.right]
        alpha = decay * alpha + prod
        beta = decay * beta + 1.0
        synch = alpha / torch.sqrt(beta + self.eps)
        return alpha, beta, synch


class CTMSyncModule(nn.Module):
    def __init__(
        self,
        D: int,
        act_left: torch.Tensor,
        act_right: torch.Tensor,
        out_left: torch.Tensor,
        out_right: torch.Tensor,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.action = RecursivePairSynchronizer(D, act_left, act_right, eps=eps)
        self.output = RecursivePairSynchronizer(D, out_left, out_right, eps=eps)

    def set_no_decay_init(self) -> None:
        self.action.set_no_decay_init()
        self.output.set_no_decay_init()


class NLM(nn.Module):
    """
    Neuron-Level Model using CTM's SuperLinear building block.
    """

    def __init__(
        self,
        *,
        D: int,
        M: int,
        deep: bool,
        H: int,
        do_layernorm_nlm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if deep:
            self.net = nn.Sequential(
                SuperLinear(in_dims=M, out_dims=2 * H, N=D, do_norm=do_layernorm_nlm, dropout=dropout),
                nn.GLU(dim=-1),
                SuperLinear(in_dims=H, out_dims=2, N=D, do_norm=do_layernorm_nlm, dropout=dropout),
                nn.GLU(dim=-1),
            )
        else:
            self.net = nn.Sequential(
                SuperLinear(in_dims=M, out_dims=2, N=D, do_norm=do_layernorm_nlm, dropout=dropout),
                nn.GLU(dim=-1),
            )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A: [B,D,M] -> [B,D,1] after GLU -> squeeze -> [B,D]
        z = self.net(A)
        if z.ndim != 3 or z.shape[-1] != 1:
            raise RuntimeError(f"Unexpected NLM output shape={tuple(z.shape)}")
        return z.squeeze(-1)


class CTMCore(nn.Module):
    """
    EEG CTM core that operates on token sequences (KV) from the tokenizer.

    Forward:
      kv_tokens: [B, N, d_kv]
    Returns:
      logits_ticks: [B, C, T_internal]
      certainty:   [B, T_internal]  (1 - normalized entropy)
      z_ticks:     [B, T_internal, D]
    """

    def __init__(self, cfg: CTMCoreConfig, *, d_kv: int, pair_bank: "torch.nn.Module") -> None:
        super().__init__()
        self.cfg = cfg
        D = int(cfg.D)
        d_input = int(cfg.d_input)

        # KV projector (Tokenizer d_kv -> CTM attention dim)
        if cfg.kv_projector == "identity":
            if int(d_kv) != d_input:
                raise ValueError(f"kv_projector='identity' requires d_kv==d_input ({d_kv} vs {d_input})")
            self.kv_projector = nn.Identity()
        elif cfg.kv_projector == "linear":
            self.kv_projector = nn.Linear(int(d_kv), d_input)
        elif cfg.kv_projector == "mlp":
            h = int(cfg.kv_mlp_hidden)
            self.kv_projector = nn.Sequential(nn.Linear(int(d_kv), h), nn.GELU(), nn.Linear(h, d_input))
        else:
            raise ValueError(cfg.kv_projector)

        # Sync modules (pairs are fixed buffers in pair_bank)
        self.sync = CTMSyncModule(
            D,
            act_left=pair_bank.act_left,
            act_right=pair_bank.act_right,
            out_left=pair_bank.out_left,
            out_right=pair_bank.out_right,
        )
        if cfg.no_decay_init:
            self.sync.set_no_decay_init()

        self.q_proj = nn.Linear(int(pair_bank.act_left.numel()), d_input)
        self.attn = nn.MultiheadAttention(d_input, int(cfg.n_heads), dropout=float(cfg.attn_dropout), batch_first=True)

        self.init = CTMInit(D, int(cfg.M_hist), mode=cfg.init_mode)
        self.norm = CTMNorm(D, d_input, ln_on_preact_hist=cfg.ln_on_preact_hist)
        self.fusion = Fusion(D, d_input, mode=cfg.fusion)

        # Synapse: shared recurrent update producing pre-activations
        self.synapse = nn.Sequential(
            nn.Dropout(float(cfg.synapse_dropout)),
            nn.Linear(D + d_input, int(cfg.synapse_hidden)),
            nn.GELU(),
            nn.Dropout(float(cfg.synapse_dropout)),
            nn.Linear(int(cfg.synapse_hidden), D),
        )

        self.nlm = NLM(
            D=D,
            M=int(cfg.M_hist),
            deep=bool(cfg.deep_nlms),
            H=int(cfg.memory_hidden_dims),
            do_layernorm_nlm=bool(cfg.do_layernorm_nlm),
            dropout=float(cfg.nlm_dropout),
        )

        # Head projects output synchronisation to logits
        Dout = int(pair_bank.out_left.numel())
        if cfg.head_type == "linear":
            self.head = nn.Linear(Dout, int(cfg.num_classes))
        elif cfg.head_type == "2fc":
            self.head = nn.Sequential(
                nn.Linear(Dout, int(cfg.head_hidden)),
                nn.GELU(),
                nn.Dropout(float(cfg.synapse_dropout)),
                nn.Linear(int(cfg.head_hidden), int(cfg.num_classes)),
            )
        else:
            raise ValueError(cfg.head_type)

    @staticmethod
    def certainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        Certainty = 1 - normalized entropy (design.md §2 Certainty 定义).
        """
        C = logits.shape[-1]
        p = torch.softmax(logits, dim=-1).clamp_min(1e-8)
        H = -(p * torch.log(p)).sum(dim=-1)  # [B]
        Hn = H / float(torch.log(torch.tensor(float(C), device=logits.device)))
        return 1.0 - Hn

    def forward(self, kv_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kv_tokens.ndim != 3:
            raise ValueError(f"Expected kv_tokens [B,N,d_kv], got shape={tuple(kv_tokens.shape)}")
        B = int(kv_tokens.shape[0])
        device = kv_tokens.device
        dtype = kv_tokens.dtype

        kv = self.kv_projector(kv_tokens)  # [B,N,d_input]

        z, A = self.init.make(B, device=device, dtype=dtype)  # z:[B,D], A:[B,D,M]

        alpha_a, beta_a, synch_a = self.sync.action.init_state(z)
        alpha_o, beta_o, synch_o = self.sync.output.init_state(z)

        T_internal = int(self.cfg.T_internal)
        C = int(self.cfg.num_classes)
        D = int(self.cfg.D)
        logits_ticks = torch.empty((B, C, T_internal), device=device, dtype=dtype)
        certainty = torch.empty((B, T_internal), device=device, dtype=dtype)
        z_ticks = torch.empty((B, T_internal, D), device=device, dtype=dtype)

        for t in range(T_internal):
            q = self.q_proj(synch_a).unsqueeze(1)  # [B,1,d_input]
            o, _ = self.attn(q, kv, kv, need_weights=False)
            o = o.squeeze(1)

            o = self.norm.norm_o(o)
            z_norm = self.norm.norm_z(z)

            u = self.fusion(z_norm, o)  # [B,D+d_input]
            a_t = self.synapse(u)  # [B,D]
            A = torch.cat([A[:, :, 1:], a_t.unsqueeze(-1)], dim=-1)
            A = self.norm.norm_A(A)

            z = self.nlm(A)  # [B,D]

            alpha_a, beta_a, synch_a = self.sync.action.step(z, alpha_a, beta_a)
            alpha_o, beta_o, synch_o = self.sync.output.step(z, alpha_o, beta_o)

            logits_t = self.head(synch_o)  # [B,C]
            logits_ticks[:, :, t] = logits_t
            certainty[:, t] = self.certainty_from_logits(logits_t)
            z_ticks[:, t, :] = z

        return logits_ticks, certainty, z_ticks

