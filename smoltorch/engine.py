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


class Tensor:

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
        self._backward = None  # function to compute local grad
        self.grad = (
            None if not self.requires_grad else self.backend.zeros_like(self.data)
        )

        self.shape = self.data.shape
        self.ndim = len(self.data.shape)

    def __add__(self, other) -> "Tensor":

        other = other if isinstance(other, Tensor) else Tensor(other)

        out_data = self.data + other.data
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:
            out._ctx = (self, other)

            def _backward():
                if self.requires_grad:
                    # TODO: implement broadcasting
                    self.grad = self.backend.add(self.grad, out.grad)
                if other.requires_grad:
                    other.grad = self.backend.add(other.grad, out.grad)

            out._backward = _backward

        return out

    def __repr__(self):
        return f"Tensor({self.data})"


if __name__ == "__main__":
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([1, 1, 1])

    print(a.shape)
    print(b.shape)
    c = a + b

    print(c)
    print(c.shape)
