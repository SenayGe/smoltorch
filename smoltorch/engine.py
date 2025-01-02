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


class OpContext:
    """Stores operation context for better debugging."""

    op_type: str
    inputs: Tuple["Tensor", ...]
    saved_tensors: Tuple[Any, ...] = ()
    saved_values: dict = None


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

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data  # z = x * y
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:
            out._ctx = (self, other)

            def _backward():

                # x.grad = ∂L/∂x = ∂L/∂z * ∂z/∂x = out.grad * y
                # y.grad = ∂L/∂y = ∂L/∂z * ∂z/∂y = out.grad * x
                if self.requires_grad:
                    # TODO: implement broadcasting
                    self.grad = self.backend.add(self.grad, other.data * out.grad)
                if other.requires_grad:
                    other.grad = self.backend.add(other.grad, self.data * out.grad)

            out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data  # Z = X @ Y
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:
            out._ctx = (self, other)

            def swap_last_two_axes(x):
                num_axes = len(x.shape)
                axes = list(range(num_axes))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                return self.backend.transpose(x, axes)

            def _backward():

                # ∂Z/∂X = grad_Z @ Y.T
                # ∂Z/∂Y = X.T @ grad_Z
                if self.requires_grad:
                    grad_x = self.backend.matmul(
                        out.grad, swap_last_two_axes(other.data)
                    )
                    self.grad = self.backend.add(self.grad, grad_x)
                if other.requires_grad:
                    grad_y = self.backend.matmul(
                        swap_last_two_axes(self.data), out.grad
                    )
                    other.grad = self.backend.add(other.grad, grad_y)

            out._backward = _backward

        return out

    def __repr__(self):
        return f"Tensor({self.data})"


if __name__ == "__main__":
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([2])

    # print(a.shape)
    # print(b.shape)
    c = a * b
    print(c)
    print(c.shape)

    x_data = [[1.0, 2.0], [3.0, 4.0]]
    y_data = [[5.0, 6.0], [7.0, 8.0]]

    # smolTorch
    x_st = Tensor(x_data, requires_grad=True)
    y_st = Tensor(y_data, requires_grad=True)
    z_st = x_st @ y_st

    # MLX
    x_mlx = mx.array(x_data)
    y_mlx = mx.array(y_data)

    # Define forward function for MLX
    def forward(x, y):
        return mx.sum(mx.matmul(x, y))

    # Get MLX gradients using value_and_grad
    grad_fn = mx.value_and_grad(forward, argnums=(0, 1))
    _, (dx_mlx, dy_mlx) = grad_fn(x_mlx, y_mlx)

    # Set gradient of output to ones for our implementation
    z_st.grad = BACKEND.ones_like(z_st.data)
    z_st._backward()

    # Print gradients for debugging
    print("\nMLX gradients:")
    print("X grad:", dx_mlx)
    print("Y grad:", dy_mlx)

    print("\nsmolTorch :")
    print("X grad:", x_st.grad)
    print("Y grad:", y_st.grad)
