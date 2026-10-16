"""Two-stage hybrid reference with an explicit C/D + bounded-KV contract."""

from .config import HybridConfig
from .state import CDCode, HybridCDKVState, weights_fingerprint

__all__ = ["HybridConfig", "CDCode", "HybridCDKVState", "weights_fingerprint"]
