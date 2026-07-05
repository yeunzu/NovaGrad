from ..backend import Backend
from .basetransform import BaseTransform


class RandomContrast(BaseTransform):
    def __init__(self, p: float = 0.5, limit: float = 0.2):
        super().__init__(p)
        self.limit = limit  # 예: 0.2면 0.8 ~ 1.2 사이의 값을 곱함

    def apply(self, x_batch):
        N = x_batch.shape[0]
        mask_shape = (N,) + (1,) * (len(x_batch.shape) - 1)
        
        # (1 - limit) ~ (1 + limit) 사이의 스칼라 곱 난수 생성
        factor = 1.0 + (self.backend.rand(mask_shape) * 2 - 1) * self.limit
        return x_batch * factor