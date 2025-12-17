"""
Reproducibility utilities (seed everything).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    seed = int(seed)

    # If deterministic CUDA algorithms are requested, cuBLAS requires this env var (CUDA >= 10.2).
    # Setting it here (and also in eeg_ctm.__init__) prevents RuntimeError from GEMM-based ops
    # like MultiheadAttention when deterministic algorithms are enabled.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Reduce sources of numerical variability on Ampere+.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions.
            pass
