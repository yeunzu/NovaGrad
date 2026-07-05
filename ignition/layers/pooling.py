from ..backend import Backend
from .base_layer import BaseLayer

from ..optimizers.adam import Adam
from ..initializers.he import He

from numba import njit, prange
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


@njit(parallel=True)
def _numba_max_pool_forward(x, pool_h, pool_w, stride, pad):
    N, C, H, W = x.shape
    out_h = int(1 + (H + 2*pad - pool_h) / stride)
    out_w = int(1 + (W + 2*pad - pool_w) / stride)

    if pad > 0:
        pad_x = np.full((N, C, H + 2*pad, W + 2*pad), -np.inf, dtype=x.dtype)
        pad_x[:, :, pad:pad+H, pad:pad+W] = x
    else:
        pad_x = x

    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    # 역전파 시 기울기를 흘려보낼 위치를 기억하기 위한 1차원 로컬 인덱스
    arg_max = np.zeros((N, C, out_h, out_w), dtype=np.int32)

    for n in prange(N):
        for c in prange(C):
            for i in range(out_h):
                for j in range(out_w):
                    max_val = -np.inf
                    max_idx = 0
                    for ph in range(pool_h):
                        for pw in range(pool_w):
                            val = pad_x[n, c, i*stride + ph, j*stride + pw]
                            if val > max_val:
                                max_val = val
                                max_idx = ph * pool_w + pw # 2차원 창 안에서의 1차원 위치
                    out[n, c, i, j] = max_val
                    arg_max[n, c, i, j] = max_idx

    return out, arg_max

@njit(parallel=True)
def _numba_max_pool_backward(dout, arg_max, x_shape, pool_h, pool_w, stride, pad):
    N, C, H, W = x_shape
    out_h = int(1 + (H + 2*pad - pool_h) / stride)
    out_w = int(1 + (W + 2*pad - pool_w) / stride)

    if pad > 0:
        dx_pad = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=dout.dtype)
    else:
        dx_pad = np.zeros((N, C, H, W), dtype=dout.dtype)

    # arg_max가 가리키는 위치에만 dout(기울기)을 흘려보냅니다.
    for n in prange(N):
        for c in prange(C):
            for i in range(out_h):
                for j in range(out_w):
                    max_idx = arg_max[n, c, i, j]
                    ph = max_idx // pool_w
                    pw = max_idx % pool_w
                    dx_pad[n, c, i*stride + ph, j*stride + pw] += dout[n, c, i, j]

    if pad > 0:
        dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    else:
        dx = dx_pad

    return dx


class Pooling(BaseLayer):
    def __init__(self, pool_h, pool_w, stride=1, pad=0, 
                 backend: Backend = None, **opt_params):
        super().__init__(backend=backend, **opt_params)

        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.x_shape = None
        self.arg_max = None # Numba 모드에서 역전파용

        # JIT 컴파일 순수 함수 매핑
        if self.backend._name == 'mlx':
            self._forward_pure = self.backend.backend_jit(self._forward_fn_mlx)
            self._backward_pure = self.backend.backend_jit(self._backward_vjp_mlx)
        elif self.backend._name == 'jax':
            self._forward_pure = self.backend.backend_jit(self._forward_fn_jax)
            self._backward_pure = self.backend.backend_jit(self._backward_vjp_jax)

    def _forward_fn_jax(self, x):
        pad_width = ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad))
        pad_x = self.backend.pad(x, pad_width=pad_width, mode='constant', constant_values=-float('inf'))
        N, C, pad_H, pad_W = pad_x.shape

        def compute_pixel(x_single, y, x_pos):
            # Conv와 달리 가중치(W)가 없고, 채널 전체를 한 번에 슬라이싱합니다.
            patch = self.backend.jax_dynamic_slice(x_single, (0, y * self.stride, x_pos * self.stride), (C, self.pool_h, self.pool_w))
            return jnp.max(patch, axis=(1, 2)) # H, W 축에 대해서만 최대값 도출 -> 결과 모양 (C,)

        v_pixel = self.backend.vmap(compute_pixel, in_axes=(None, None, 0))
        v_row = self.backend.vmap(v_pixel, in_axes=(None, 0, None))
        v_batch = self.backend.vmap(v_row, in_axes=(0, None, None))

        y_idx = self.backend.arange(0, self.out_h, 1, self.backend.dtype('int32'))
        x_idx = self.backend.arange(0, self.out_w, 1, self.backend.dtype('int32'))

        out = v_batch(pad_x, y_idx, x_idx) # vmap의 결과는 (N, out_h, out_w, C)로 나옵니다.
        return jnp.transpose(out, (0, 3, 1, 2)) # 다시 (N, C, out_h, out_w)로 변환

    def _forward_fn_mlx(self, x):
        pad_width = ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad))
        pad_x = self.backend.pad(x, pad_width=pad_width, mode='constant', constant_values=-float('inf'))
        N, C, pad_H, pad_W = pad_x.shape

        def compute_pixel(x_single, y, x_pos):
            c_idx = self.backend.arange(0, C, 1, self.backend.dtype('int32'))[:, None, None]
            y_idx = (y * self.stride) + self.backend.arange(0, self.pool_h, 1, self.backend.dtype('int32'))[None, :, None]
            x_idx = (x_pos * self.stride) + self.backend.arange(0, self.pool_w, 1, self.backend.dtype('int32'))[None, None, :]

            patch = x_single[c_idx, y_idx, x_idx]
            return self.backend.max(patch, axis=(1, 2)) # H, W 축에 대해서만 최대값

        v_pixel = self.backend.vmap(compute_pixel, in_axes=(None, None, 0))
        v_row = self.backend.vmap(v_pixel, in_axes=(None, 0, None))
        v_batch = self.backend.vmap(v_row, in_axes=(0, None, None))

        y_idx = self.backend.arange(0, self.out_h, 1, self.backend.dtype('int32'))
        x_idx = self.backend.arange(0, self.out_w, 1, self.backend.dtype('int32'))

        out = v_batch(pad_x, y_idx, x_idx) # (N, out_h, out_w, C)
        return self.backend.transpose(out, (0, 3, 1, 2))

    def forward(self, x, **kwargs):
        N, C, H, W = x.shape
        self.x = x
        self.x_shape = x.shape
        self.out_h = int(1 + (H + 2 * self.pad - self.pool_h) / self.stride)
        self.out_w = int(1 + (W + 2 * self.pad - self.pool_w) / self.stride)

        if self.backend._name == 'mlx' or self.backend._name == 'jax':
            out = self._forward_pure(x)
        else:
            # numpy/cpu의 경우
            out, self.arg_max = _numba_max_pool_forward(x, self.pool_h, self.pool_w, self.stride, self.pad)
            
        return out

    def _backward_vjp_jax(self, x, dout):
        # 파라미터가 없으므로 x만 추적합니다.
        out, vjp_fun = jax.vjp(self._forward_fn_jax, x)
        dx, = vjp_fun(dout) # 튜플 언패킹
        return dx

    def _backward_vjp_mlx(self, x, dout):
        outs, vjps = mx.vjp(self._forward_fn_mlx, [x], [dout])
        dx = vjps[0] # 리스트에서 첫 번째 원소(dx) 추출
        return dx

    def backward(self, dout, **kwargs):
        if self.backend._name == 'mlx':
            dx = self._backward_vjp_mlx(self.x, dout)
        elif self.backend._name == 'jax':
            dx = self._backward_vjp_jax(self.x, dout)
        else:
            dx = _numba_max_pool_backward(
                dout, self.arg_max, self.x_shape, self.pool_h, self.pool_w, self.stride, self.pad
            )
            
        self.x = None # 메모리 절약
        self.arg_max = None
        return dx
    
    # update는 부모 클래스 사용

    def output_shape(self, input_shape):
        C, H, W = input_shape
        out_h = int(1 + (H + 2 * self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - self.pool_w) / self.stride)
        return (C, out_h, out_w)
    
    # astype은 부모 클래스 사용