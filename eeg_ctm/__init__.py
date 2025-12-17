"""
EEG-CTM: Continuous Thought Machine for cross-subject MI EEG classification.
"""

from __future__ import annotations

# NOTE: When `torch.use_deterministic_algorithms(True)` is enabled on CUDA>=10.2,
# cuBLAS requires this environment variable for deterministic GEMMs.
# We set a safe default early (before importing torch in submodules).
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
