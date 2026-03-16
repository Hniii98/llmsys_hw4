"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        
        # COPY FROM ASSIGN2_3
        weights_np = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.weights = Parameter(tensor_from_numpy(weights_np, backend=backend))
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        
        # COPY FROM ASSIGN2_3
        x = one_hot(x, self.num_embeddings)
        w = self.weights.value
        x = x.contiguous().view(bs*seq_len, self.num_embeddings)
        out = x @ w
        out = out.view(bs, seq_len, self.embedding_dim)
        return out

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        # COPY FROM ASSIGN2_3
        if self.training == False or self.p_dropout == 0:
            return x

        mask = np.random.binomial(1, 1-self.p_dropout, size=x.shape).astype(np.float32)
        m = tensor_from_numpy(mask, backend=x.backend)
        out = (x * m) / (1 - self.p_dropout)
        return out


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weight - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
        """
        self.out_size = out_size
        self.in_size = in_size
        # COPY FROM ASSIGN2_3
        self.weights = self._init_param(in_size, out_size, backend=backend)
        if bias:
            self.bias = self._init_param(out_size, backend=backend)
        else:
            self.bias = None

    def _init_param(self, *shape, backend: TensorBackend):
        # Uniform(-1/sqrt(in_size), 1/sqrt(in_size))
        inv = 1.0 / (self.in_size ** 0.5)
        r = 2 * (rand(shape, backend=backend) - 0.5) * inv
        return Parameter(r)
    
    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        
        # COPY FROM ASSIGN2_3
        out = x @ self.weights.value
        if self.bias is not None:
            out = out + self.bias.value
        return out


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        
        # COPY FROM ASSIGN2_3
        self.weights = Parameter(ones(shape=(dim,), backend=backend))
        self.bias = Parameter(zeros(shape=(dim,), backend=backend))

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        
        # COPY FROM ASSIGN2_3
        m = x.mean(dim=1)
        v = x.var(dim=1)
        b = self.bias.value
        w = self.weights.value
        inv = 1.0 / ((v + self.eps) ** 0.5)
        out = w * ((x - m) * inv) + b
        return out
