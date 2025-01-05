import platform
from contextlib import contextmanager
from typing import Any, Tuple

import numpy as np

# Check if running on Apple Silicon
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


class OpContext:
    """Stores operation info and values needed for backward pass."""

    def __init__(self):
        self.op: str = None
        self.inputs: Tuple["Tensor", ...] = None  # for gradient accumulation
        self.saved_tensors: Tuple[Any, ...] = ()  # for computing gradients


class Function:
    @staticmethod
    def forward(ctx: OpContext, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: OpContext, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        # If gradient computation is disabled, just run forward pass
        if not Tensor._compute_grad:
            return cls.forward(OpContext(), *args)

        ctx = OpContext()
        ctx.op = cls.__name__
        ctx.inputs = args

        # Run forward pass
        output = cls.forward(ctx, *args)

        # Check if any input requires gradient
        requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        if requires_grad:
            output._ctx = ctx

            def _backward():
                grads = cls.backward(ctx, output.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for inp, grad in zip(ctx.inputs, grads):
                    if inp.requires_grad:
                        inp.grad = BACKEND.add(inp.grad, grad)

            output._backward = _backward

        return output


class Add(Function):
    @staticmethod
    def forward(ctx: OpContext, x: "Tensor", y: "Tensor"):
        return Tensor(x.data + y.data, requires_grad=x.requires_grad or y.requires_grad)

    @staticmethod
    def backward(ctx: OpContext, grad_output):
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: OpContext, x: "Tensor", y: "Tensor"):
        ctx.saved_tensors = (x, y)  # Need both inputs for gradient computation
        return Tensor(x.data * y.data, requires_grad=x.requires_grad or y.requires_grad)

    @staticmethod
    def backward(ctx: OpContext, grad_output):
        x, y = ctx.saved_tensors
        return y.data * grad_output, x.data * grad_output  # Uses saved x, y values


class MatMul(Function):
    @staticmethod
    def forward(ctx: OpContext, x: "Tensor", y: "Tensor"):
        ctx.saved_tensors = (x, y)  # Need both inputs for gradient computation
        return Tensor(x.data @ y.data, requires_grad=x.requires_grad or y.requires_grad)

    @staticmethod
    def backward(ctx: OpContext, grad_output):
        x, y = ctx.saved_tensors

        def swap_last_two_axes(x):
            num_axes = len(x.shape)
            axes = list(range(num_axes))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            return BACKEND.transpose(x, axes)

        # Uses saved x, y values for transpose operations
        grad_x = BACKEND.matmul(grad_output, swap_last_two_axes(y.data))
        grad_y = BACKEND.matmul(swap_last_two_axes(x.data), grad_output)

        return grad_x, grad_y


class Tensor:
    _compute_grad = True

    @classmethod
    @contextmanager
    def no_grad(cls):
        prev = cls._compute_grad
        cls._compute_grad = False
        try:
            yield
        finally:
            cls._compute_grad = prev

    def __init__(self, data: Any, requires_grad: bool = False, dtype=None):
        self.backend = BACKEND
        self.dtype = dtype or self.backend.float32

        # Convert data to array
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
        self._ctx = None  # Context for backward pass
        self.requires_grad = requires_grad and self._compute_grad
        self._backward = None
        self.grad = (
            None if not self.requires_grad else self.backend.zeros_like(self.data)
        )

        self.shape = self.data.shape
        self.ndim = len(self.data.shape)

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add.apply(self, other)

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Mul.apply(self, other)

    def __matmul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MatMul.apply(self, other)

    def __repr__(self):
        op_str = f" (op: {self._ctx.op})" if self._ctx and self._ctx.op else ""
        return f"Tensor({self.data}{op_str})"


if __name__ == "__main__":
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([2])

    # print(a.shape)
    # print(b.shape)
    c = a + b
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
