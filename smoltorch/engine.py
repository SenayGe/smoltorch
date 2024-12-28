import platform
from contextlib import contextmanager
from typing import *

import numpy as np

# if running on apple silicon
USE_MLX = platform.processor() == "arm" and platform.system() == "Darwin"

# Import appropriate backend
if USE_MLX:
    try:
        import mlx.core as mx

        BACKEND = mx
    except ImportError:
        BACKEND = np

else:
    BACKEND = np
# TODO: add option for JAX


class Tenosor:

    _compute_grad = True

    @classmethod
    @contextmanager
    def no_grad(cls):
        """Context manager to temporarily disable gradient computation"""
        prev = cls._compute_grad
        cls._compute_grad = False
        try:
            yield
        finally:
            cls._compute_grad = prev

    def __init__(self, data: Any, requires_grad: bool = False, dtype=None):
        self.backend = BACKEND
        self.dtype = dtype or self.backend.float32

        # convert data to array
        if isinstance(data, (int, float)):
            self.data = self.backend.array([data], dtype=self.dtype)
        elif isinstance(data, (list, tuple)):
            self.data = self.backend.array(data, dtype=self.dtype)
        elif isinstance(data, (np.ndarray, getattr(mx, "array", type(None)))):
            self.data = self.backend.array(data, dtype=self.dtype)
        elif isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Gradient related attributes
        self._ctx = None  # holds the operation context
        self.requires_grad = requires_grad and self._compute_grad
