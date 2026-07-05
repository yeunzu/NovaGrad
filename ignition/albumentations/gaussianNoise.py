from ..backend import Backend
from .basetransform import BaseTransform


class GaussianNoise(BaseTransform):
    def __init__(self, p: float = 0.5, mean: float = 0.0, std: float = 0.05):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def apply(self, x_batch):
        """가우시안 노이즈 더하기"""
        # backend.randn은 표준정규분포(N(0,1))를 반환하므로 std와 mean으로 스케일링
        noise = (self.backend.randn(x_batch.shape) * self.std) + self.mean
        return x_batch + noise