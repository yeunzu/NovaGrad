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
except ImportError:
    pass

# Conv를 위한 사전 함수들
# numba 쓸거면 Backend가 아닌 numpy 사용이 강제됨
@njit(parallel=True)
def _numba_conv(x, W, B, pad, stride):
    N, C, H, W_dim = x.shape
    FN, FC, FH, FW = W.shape

    out_h = int(1 + (H + 2*pad - FH) / stride)
    out_w = int(1 + (W_dim + 2*pad - FW) / stride)

    if pad > 0:
        pad_x = np.zeros((N, C, H + 2*pad, W_dim + 2*pad), dtype=x.dtype)
        pad_x[:, :, pad:pad+H, pad:pad+W_dim] = x
    else:
        pad_x = x

    out = np.zeros((N, FN, out_h, out_w), dtype=x.dtype)

    for n in prange(N):
        for f in prange(FN):
            for i in range(out_h):
                for j in range(out_w):
                    # 각 출력 칸 (n, f, i, j)에 들어갈 수식 정의
                    # x의 해당 영역 슬라이싱 & 아다마르 곱
                    value = 0.0
                    for c in range(C):
                        for kh in range(FH):
                            for kw in range(FW):
                                value += pad_x[n, c, i*stride + kh, j*stride + kw] * W[f, c, kh, kw]
                    out[n, f, i, j] = value

    # Bias 더하기
    for n in prange(N):
        for f in prange(FN):
            for i in range(out_h):
                for j in range(out_w):
                    out[n, f, i, j] += B[f]

    return out


@njit(parallel=True)
def numba_conv_backward(x, W, dout, pad, stride):
    N, C, H, W_dim = x.shape
    FN, FC, FH, FW = W.shape
    _, _, out_h, out_w = dout.shape

    # 1. dB
    dB = np.zeros(FN, dtype=dout.dtype)
    for f in range(FN):
        dB[f] = np.sum(dout[:, f, :, :])

    # 2. 패딩된 x 복구 (dW 계산용)
    if pad > 0:
        x_pad = np.zeros((N, C, H + 2*pad, W_dim + 2*pad), dtype=x.dtype)
        x_pad[:, :, pad:pad+H, pad:pad+W_dim] = x
    else:
        x_pad = x

    # 3. dW
    dW = np.zeros_like(W)
    for n in prange(N):
        for f in prange(FN):
            for c in range(C):
                for kh in range(FH):
                    for kw in range(FW):
                        for i in range(out_h):
                            for j in range(out_w):
                                dW[f, c, kh, kw] += x_pad[n, c, i*stride + kh, j*stride + kw] * dout[n, f, i, j]

    # 4. dX
    dX_pad = np.zeros_like(x_pad)
    for n in prange(N):
        for f in prange(FN):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        for kh in range(FH):
                            for kw in range(FW):
                                dX_pad[n, c, i*stride + kh, j*stride + kw] += dout[n, f, i, j] * W[f, c, kh, kw]


    # 패딩된 dx -> 원래 x 크기만큼 잘라내기
    if pad > 0:
        dX = dX_pad[:, :, pad:pad+H, pad:pad+W_dim]
    else:
        dX = dX_pad

    return dX, dW, dB


class Conv(BaseLayer):
    def __init__(self, filter_num, filter_channels, filter_height, filter_width, stride=1, 
                 backend: Backend = None, optimizer = Adam, learning_rate=0.001, initializer=He, 
                 **opt_params):
        super().__init__(backend, optimizer, learning_rate, **opt_params)

        self.stride = stride
        self.pad = self.pad
        
        self.x = None
        self.x_shape = None

        fan_in = filter_channels * filter_height * filter_width # C*H*W
        weight_2d = initializer(fan_in, filter_num, backend=self.backend) # (FC*FH*FW, FN)
        W_4d = self.backend.reshape(weight_2d, (filter_channels, filter_height, filter_width, filter_num))
        W_4d = self.backend.transpose(W_4d, (3, 0, 1, 2))

        self.params = [
            W_4d, 
            self.backend.zeros(filter_num)
        ]
        self.grads = [None, None]

        if self.backend._name == 'mlx':
            self._forward_pure = self.backend.backend_jit(self._forward_fn_mlx)
            self.backward_pure = self.backend.backend_jit(self._backward_vjp_mlx)
        elif self.backend._name == 'jax':
            self._forward_pure = self.backend.backend_jit(self._forward_fn_jax)
            self._backward_pure = self.backend.backend_jit(self._backward_vjp_jax)

    def _forward_fn_jax(self, x, W, B):
        # pad_x = self.backend.pad(x, self.pad)
        # 4차원 (N, C, H, W) 기준 패딩 너비 설정
        pad_width = ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad))
        pad_x = self.backend.pad(x, pad_width=pad_width, mode='constant')
        N, C, pad_H, pad_W = pad_x.shape
        FN, FC, FH, FW = W.shape

        out_h = self.out_h
        out_w = self.out_w

        def compute_pixel(x_single, w_single, y, x_pos):
            # patch = jax.lax.dynamic_slice(x_single, (0, y * self.stride, x_pos * self.stride), (C, FH, FW))
            patch = self.backend.jax_dynamic_slice(x_single, (0, y * self.stride, x_pos * self.stride), (C, FH, FW))
            # return jnp.sum(patch * w_single)
            return self.backend.sum(patch * w_single)

        # vmap 중첩 -> 차원(W, H, FN, N) 확장
        # v_pixel = jax.vmap(compute_pixel, in_axes=(None, None, None, 0))
        # v_row = jax.vmap(v_pixel, in_axes=(None, None, 0, None))
        # v_filter = jax.vmap(v_row, in_axes=(None, 0, None, None))
        # v_batch = jax.vmap(v_filter, in_axes=(0, None, None, None))
        v_pixel = self.backend.vmap(compute_pixel, in_axes=(None, None, None, 0))
        v_row = self.backend.vmap(v_pixel, in_axes=(None, None, 0, None))
        v_filter = self.backend.vmap(v_row, in_axes=(None, 0, None, None))
        v_batch = self.backend.vmap(v_filter, in_axes=(0, None, None, None))

        y_idx = self.backend.arange(0, out_h, 1, self.backend.dtype('int32'))
        x_idx = self.backend.arange(0, out_w, 1, self.backend.dtype('int32'))

        out = v_batch(pad_x, W, y_idx, x_idx)
        return out + B.reshape(1, FN, 1, 1)

    def _forward_fn_mlx(self, x, W, B):
        # pad_x = self.backend.pad(x, self.pad)
        # 4차원 (N, C, H, W) 기준 패딩 너비 설정
        pad_width = ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad))
        pad_x = self.backend.pad(x, pad_width=pad_width, mode='constant')
        N, C, pad_H, pad_W = pad_x.shape
        FN, FC, FH, FW = W.shape

        out_h = self.out_h
        out_w = self.out_w

        def compute_pixel(x_single, w_single, y, x_pos):
            c_idx = self.backend.arange(0, C, 1, self.backend.dtype('int32'))[:, None, None]
            y_idx = (y * self.stride) + self.backend.arange(0, FH, 1, self.backend.dtype('int32'))[None, :, None]
            x_idx = (x_pos * self.stride) + self.backend.arange(0, FW, 1, self.backend.dtype('int32'))[None, None, :]

            patch = x_single[c_idx, y_idx, x_idx]
            return self.backend.sum(patch * w_single)

        v_pixel = self.backend.vmap(compute_pixel, in_axes=(None, None, None, 0))
        v_row = self.backend.vmap(v_pixel, in_axes=(None, None, 0, None))
        v_filter = self.backend.vmap(v_row, in_axes=(None, 0, None, None))
        v_batch = self.backend.vmap(v_filter, in_axes=(0, None, None, None))

        y_idx = self.backend.arange(0, out_h, 1, self.backend.dtype('int32'))
        x_idx = self.backend.arange(0, out_w, 1, self.backend.dtype('int32'))

        out = v_batch(pad_x, W, y_idx, x_idx)
        return out + B.reshape(1, FN, 1, 1)
    
    def forward(self, x, **kwargs):
        N, C, H, W = x.shape
        self.x_shape = x.shape
        FN, FC, FH, FW = self.params[0].shape
        self.out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        self.out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        if self.backend._name == 'mlx' or self.backend._name == 'jax':
            out = self._forward_pure(x, self.params[0], self.params[1])
        else:
            # numpy의 경우
            out = _numba_conv(x, self.params[0], self.params[1], pad=self.pad, stride=self.stride)
        self.x = x
        return out
    
    def _backward_vjp_jax(self, x, W, B, dout):
        out, vjp_fun = jax.vjp(self._forward_fn_jax, x, W, B)
        dx, dW, dB = vjp_fun(dout)
        return dx, dW, dB
    
    def _backward_vjp_mlx(self, x, W, B, dout):
        outs, vjps = mx.vjp(self._forward_fn_mlx, [x, W, B], [dout])
        dx, dW, dB = vjps
        return dx, dW, dB
    
    def backward(self, dout, **kwargs):
        if self.backend._name == 'mlx':
            dx, dw, db = self._backward_vjp_mlx(self.x, self.params[0], self.params[1], dout)
        elif self.backend._name == 'jax':
            dx, dw, db = self._backward_vjp_jax(self.x, self.params[0], self.params[1], dout)
        else:
            dx, dw, db = numba_conv_backward(
                self.x, self.params[0], dout, pad=self.pad, stride=self.stride
            )

        self.grads[0] = dw
        self.grads[1] = db
        return dx
    
    # update는 부모 클래스 사용

    def output_shape(self, input_shape):
        # input_shape: (C, H, W)
        C, H, W = input_shape
        FN, _, FH, FW = self.params[0].shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)
        return (FN, out_h, out_w)
    
    # astype는 부모 클래스 사용