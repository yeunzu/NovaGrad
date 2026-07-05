from ..backend import Backend
from .basetransform import BaseTransform


class RandomBrightness(BaseTransform):
    def __init__(self, p: float = 0.5, limit: float = 0.2):
        super().__init__(p)
        self.limit = limit  # 예: 0.2면 -0.2 ~ 0.2 사이의 값을 더함

    def apply(self, x_batch):
        # N 사이즈의 난수 배열 생성 후 (N, 1, 1, 1) 로 형태 맞추기
        N = x_batch.shape[0]
        mask_shape = (N,) + (1,) * (len(x_batch.shape) - 1)
        
        # -limit ~ +limit 사이의 균등 분포 난수 생성
        # backend.rand()는 0~1 사이이므로 스케일링 수행
        factor = (self.backend.rand(mask_shape) * 2 - 1) * self.limit
        return x_batch + factor