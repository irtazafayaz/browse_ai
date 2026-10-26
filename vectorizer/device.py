"""
Pick the torch device all local models run on.

Order: EMBED_DEVICE env override -> Apple MPS -> CUDA -> CPU. The bigger models
(bge-large, SigLIP) are only comfortable on an accelerator; on a CPU-only box
(Docker/Railway) point the *_MODEL env vars at the small variants instead.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)


def pick_device() -> str:
    forced = os.environ.get("EMBED_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()
logger.info("Model device: %s", DEVICE)

# Several of the chosen checkpoints ship fp16 weights, which transformers keeps
# as-is. Fine (and faster) on MPS/CUDA; on CPU fp16 matmuls are slow or
# unsupported, so upcast there. EMBED_DTYPE=float32 forces the upcast anywhere.
_FORCE_DTYPE = os.environ.get("EMBED_DTYPE")
MODEL_KWARGS = (
    {"dtype": getattr(torch, _FORCE_DTYPE)} if _FORCE_DTYPE
    else {"dtype": torch.float32} if DEVICE == "cpu"
    else {}
)
