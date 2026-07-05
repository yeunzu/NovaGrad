import numpy as np
from random import randint

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

_DTYPE_MAP = {
    'numpy': {
        'FP32': np.float32,
        'FP16': np.float16,
        'BF16': np.float32,  # numpy는 bf16 미지원 → fp32 fallback
        'INT32': np.int32,
        'INT64': np.int64,
        'INT8' : np.int8, 
        'UINT8' : np.uint8, 
        'UINT32' : np.uint32, 
    },
    'mlx': {
        'FP32': mx.float32       if MLX_AVAILABLE else None,
        'FP16': mx.float16       if MLX_AVAILABLE else None,
        'BF16': mx.bfloat16      if MLX_AVAILABLE else None,
        'INT32': mx.int32        if MLX_AVAILABLE else None,
        'INT64': mx.int64        if MLX_AVAILABLE else None,  # mlx는 int32까지만 지원, 주의
        'INT8' : mx.int8         if MLX_AVAILABLE else None, 
        'UINT8' : mx.uint8       if MLX_AVAILABLE else None, 
        'UINT32' : mx.uint32     if MLX_AVAILABLE else None, 
    } if MLX_AVAILABLE else {},
    'jax': {
        'FP32': jnp.float32      if JAX_AVAILABLE else None,
        'FP16': jnp.float16      if JAX_AVAILABLE else None,
        'BF16': jnp.bfloat16     if JAX_AVAILABLE else None,
        'INT32': jnp.int32       if JAX_AVAILABLE else None,
        'INT64': jnp.int64       if JAX_AVAILABLE else None,
        'INT8' : jnp.int8        if JAX_AVAILABLE else None, 
        'UINT8' : jnp.uint8      if JAX_AVAILABLE else None, 
        'UINT32' : jnp.uint32    if JAX_AVAILABLE else None,   
    } if JAX_AVAILABLE else {},
}


class Backend:
    """
    사용 가능한 하드웨어를 자동으로 탐지해 백엔드를 선택.
    우선순위: Apple(MLX) > TPU (Jax) > GPU (Jax) > CPU (Numpy)
    """
    def __init__(self):
        self.be = None
        self._name = None

        self.apple_status = False

        gpu_status = False
        tpu_status = False

        self.JAX_AVAILABLE = JAX_AVAILABLE
        self.MLX_AVAILABLE = MLX_AVAILABLE

        # 1. apple device check
        if MLX_AVAILABLE:
            try:
                apple_device = mx.default_device()
                if apple_device.type == mx.DeviceType.gpu:
                    self.apple_status = True
            except Exception:
                pass
        
        # 2. gpu/tpu device check
        if JAX_AVAILABLE:
            try:
                jax_devices = jax.devices()
                for device in jax_devices:
                    platform = device.platform
                    if platform == 'tpu':
                        tpu_status = True
                    elif platform == 'gpu':
                        gpu_status = True
                start_seed = randint(0, 2**32 - 1)
                self.jax_key = jax.random.PRNGKey(start_seed)
            except Exception:
                pass

        if self.apple_status:
            self.be = mx
            self._name = 'mlx'
        elif (tpu_status or gpu_status) and not self.apple_status:
            self.be = jnp
            self._name = 'jax'
        elif self.JAX_AVAILABLE:
            self.be = jnp
            self._name = 'jax'
        else:
            self.be = np
            self._name = 'numpy'

    def now_backend(self) -> str:
        return self._name
    
    def dtype(self, dtype_str: str):
        """
        'fp32', 'fp16', 'bf16', 'int32', 'int64' 등의 문자열을
        현재 백엔드에 맞는 dtype 객체로 변환한다.
        """
        dtype_str = dtype_str.upper()
        table = _DTYPE_MAP.get(self._name, {})
        if (dtype_str not in table) or (table[dtype_str] is None):
            raise ValueError(f"Backend '{self._name}'은 dtype '{dtype_str}'을 지원하지 않습니다.")
        return table[dtype_str]

    def cast(self, array, dtype_str: str):
        """
        배열을 지정한 dtype로 변환한다.
        """
        dt = self.dtype(dtype_str)
        if self._name == 'mlx':
            return array.astype(dt)
        elif self._name == 'jax':
            return array.astype(dt)
        else:
            return array.astype(dt)


    ## 연산 메서드
    def array(self, x, dtype=None):
        if self._name == 'mlx':
            return mx.array(x, dtype=dtype)
        elif self._name == 'jax':
            return jnp.array(x, dtype=dtype)
        else:
            return np.array(x, dtype=dtype)
        
    def zeros(self, shape, dtype=None):
        if self._name == 'mlx':
            return mx.zeros(shape, dtype=dtype or mx.float32)
        elif self._name == 'jax':
            return jnp.zeros(shape, dtype=dtype or jnp.float32)
        else:
            return np.zeros(shape, dtype=dtype or np.float32)
        
    def zeros_like(self, x):
        if self._name == 'mlx':
            return mx.zeros_like(x)
        elif self._name == 'jax':
            return jnp.zeros_like(x)
        else:
            return np.zeros_like(x)
        
    def ones(self, shape, dtype=None):
        if self._name == 'mlx':
            return mx.ones(shape, dtype=dtype or mx.float32)
        elif self._name == 'jax':
            return jnp.ones(shape, dtype=dtype or jnp.float32)
        else:
            return np.ones(shape, dtype=dtype or np.float32)

    def arange(self, start, stop, step, dtype):
        if self._name == 'mlx':
            return mx.arange(start=start, stop=stop, step=step, dtype=dtype or mx.float32)
        elif self._name == 'jax':
            return jnp.arange(start=start, stop=stop, step=step, dtype=dtype or jnp.float32)
        else:
            return np.arange(start=start, stop=stop, step=step, dtype=dtype or np.float32)

    def randn(self, shape):
        if self._name == 'mlx':
            return mx.random.normal(shape=shape)
        elif self._name == 'jax':
            current_key, next_key = jax.random.split(self.jax_key, 2)
            self.jax_key = next_key
            return jax.random.normal(current_key, shape=shape)
        else:
            return np.random.randn(*shape)

    def rand(self, shape):
        if self._name == 'mlx':
            return mx.random.uniform(shape=shape)
        elif self._name == 'jax':
            current_key, next_key = jax.random.split(self.jax_key, 2)
            self.jax_key = next_key
            return jax.random.uniform(key=current_key, shape=shape)
        else:
            return np.random.rand(*shape)

    def randint(self, shape=None):
        assert shape!=None, "shape 파라미터가 입력되지 않았습니다."
        if self._name == 'mlx':
            return mx.random.randint(shape=shape)
        elif self._name == 'jax':
            current_key, next_key = jax.random.split(self.jax_key, 2)
            self.jax_key = next_key
            return jax.random.randint(key=current_key, shape=shape)
        else:
            return np.random.randint(size=shape)
        
        
    def pad(self, array, pad_width, mode, **kwargs):
        if self._name == 'mlx':
            return mx.pad(array, pad_width=pad_width, mode=mode, **kwargs)
        elif self._name == 'jax':
            return jnp.pad(array, pad_width=pad_width, mode=mode, **kwargs)
        else:
            return np.pad(array, pad_width=pad_width, mode=mode, **kwargs)

    def copy(self, array):
        """새 객체 생성용 메서드. mlx가 copy를 지원하지 않는 관계로 꼼수."""
        if self._name == 'mlx':
            return mx.array(array)
        elif self._name == 'jax':
            return jnp.copy(array)
        else:
            return np.copy(array)

    def where(self, condition, x, y):
        if self._name == 'mlx':
            return mx.where(condition, x, y)
        elif self._name == 'jax':
            return jnp.where(condition, x, y)
        else:
            return np.where(condition, x, y)

    def backend_jit(self, fun):
        if self._name == 'mlx':
            return mx.compile(fun)
        elif self._name == 'jax':
            return jax.jit(fun)
        else:
            try:
                from numba import njit
                return njit(fun)
            except ImportError:
                print("[Backend] 경고: numba 모듈이 없어 jit 컴파일 없이 함수를 실행합니다.\n")
                return fun

    def vmap(self, fun, in_axes=0, out_axes=0):
        if self._name == 'mlx':
            return mx.vmap(fun=fun, in_axes=in_axes, out_axes=out_axes)
        elif self._name == 'jax':
            return jax.vmap(fun=fun, in_axes=in_axes, out_axes=out_axes)
        else:
            return AttributeError(f"{self._name}은 vmap 메서드가 없습니다.")

    def jax_dynamic_slice(self, operand, start_indices, slice_sizes):
        """jax의 jax.lax.dynamic_slice 메서드를 Backend 클래스로 사용할 수 있게 하기 위함"""
        if self._name == 'jax':
            return jax.lax.dynamic_slice(operand=operand, start_indices=start_indices, slice_sizes=slice_sizes)
        else:
            return AttributeError(f"{self._name}은 vmap 메서드가 없습니다.")

    def dynamic_slice(self, operand, start_indices, slice_sizes):
        """
        주어진 시작 인덱스와 크기를 기반으로 동적 슬라이싱을 수행.
        모든 백엔드에서 동일한 결과를 반환하도록 추상화.
        """
        if self._name == 'jax':
            return jax.lax.dynamic_slice(
                operand=operand,
                start_indices=start_indices,
                slice_sizes=slice_sizes
            )
        else:
            # mlx와 numpy는 파이썬 내장 slice 객체를 튜플로 만들어 인덱싱에 사용
            # (start, start+size) 형태의 슬라이스를 차원별로 생성
            slices = tuple(
                slice(int(start), int(start + size)) for start, size in zip(start_indices, slice_sizes)
            )
            return operand[slices]

    def matmul(self, a, b):
        """행렬 곱, np.dot(2D) 대응"""
        if self._name == 'mlx':
            return mx.matmul(a, b)
        elif self._name == 'jax':
            return jnp.matmul(a, b)
        else:
            return np.dot(a, b)

    def multiply(self, a, b):
        """원소 곱"""
        if self._name == 'mlx':
            return mx.multiply(a, b)
        elif self._name == 'jax':
            return jnp.multiply(a, b)
        else:
            return np.multiply(a, b)

    def dot(self, a, b):
        """matmul의 별칭, np.dot과 동일하게 사용 가능."""
        return self.matmul(a, b)
    
    def prod(self, arr, axis=None, keepdims=False):
        if self._name == 'mlx':
            return mx.prod(a=arr, axis=axis, keepdims=keepdims)
        elif self._name == 'jax':
            return jnp.prod(a=arr, axis=axis, keepdims=keepdims)
        else:
            return np.prod(a=arr, axis=axis, keepdims=keepdims)

    def sum(self, x, axis=None, keepdims=False):
        if self._name == 'mlx':
            return mx.sum(x, axis=axis, keepdims=keepdims)
        elif self._name == 'jax':
            return jnp.sum(x, axis=axis, keepdims=keepdims)
        else:
            return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        if self._name == 'mlx':
            return mx.mean(x, axis=axis, keepdims=keepdims)
        elif self._name == 'jax':
            return jnp.mean(x, axis=axis, keepdims=keepdims)
        else:
            return np.mean(x, axis=axis, keepdims=keepdims)
        
    def max(self, x, axis=None, keepdims=False):
        if self._name == 'mlx':
            return mx.max(x, axis=axis, keepdims=keepdims)
        elif self._name == 'jax':
            return jnp.max(x, axis=axis, keepdims=keepdims)
        else:
            return np.max(x, axis=axis, keepdims=keepdims)
        
    def argmax(self, x, axis=None):
        if self._name == 'mlx':
            return mx.argmax(x, axis=axis)
        elif self._name == 'jax':
            return jnp.argmax(x, axis=axis)
        else:
            return np.argmax(x, axis=axis)

    def argmin(self, x, axis=None):
        if self._name == 'mlx':
            return mx.argmin(x, axis=axis)
        elif self._name == 'jax':
            return jnp.argmin(x, axis=axis)
        else:
            return np.argmin(x, axis=axis)
        
    def exp(self, x):
        if self._name == 'mlx':
            return mx.exp(x)
        elif self._name == 'jax':
            return jnp.exp(x)
        else:
            return np.exp(x)
        
    def erf(self, array):
        if self._name == 'mlx':
            import mlx.core as mx
            return mx.erf(array)
        elif self._name == 'jax':
            from jax.scipy.special import erf
            return erf(array)
        else:
            from scipy.special import erf
            return erf(array)

    def log(self, x):
        if self._name == 'mlx':
            return mx.log(x)
        elif self._name == 'jax':
            return jnp.log(x)
        else:
            return np.log(x)
        
    def sqrt(self, x):
        if self._name == 'mlx':
            return mx.sqrt(x)
        elif self._name == 'jax':
            return jnp.sqrt(x)
        else:
            return np.sqrt(x)
        
    def maximum(self, x, val):
        if self._name == 'mlx':
            return mx.maximum(x, val)
        elif self._name == 'jax':
            return jnp.maximum(x, val)
        else:
            return np.maximum(x, val)

    def shape(self, arr):
        # if self._name == 'mlx':
        #     return arr.shape
        # elif self._name == 'jax':
        #     return arr.shape
        # else:
        #     return arr.shape
        return arr.shape # 모든 환경에서 같은 명령어(같은 변수명)로 같은 동작을 하므로 굳이 백엔드를 구별하지 않음
        
    def reshape(self, x, shape):
        if self._name == 'mlx':
            return mx.reshape(x, shape)
        elif self._name == 'jax':
            return jnp.reshape(x, shape)
        else:
            return np.reshape(x, shape)
        
    def transpose(self, x, axes=None):
        if self._name == 'mlx':
            return mx.transpose(x, axes)
        elif self._name == 'jax':
            return jnp.transpose(x, axes)
        else:
            return np.transpose(x, axes)

    def to_numpy(self, x) -> np.ndarray:
        """현재 백엔드 배열을 numpy 배열로 변환"""
        if self._name != 'numpy':
            return np.array(x)
        else:
            return x
        
