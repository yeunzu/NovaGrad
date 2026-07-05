from ..backend import Backend
from .basetransform import BaseTransform


class HorizontalFlip(BaseTransform):
    def __init__(self, p: float = 0.5, input_format: str = 'NCHW'):
        super().__init__(p)
        self.input_format = input_format

    def apply(self, x_batch):
        """배열 슬라이싱을 이용한 초고속 좌우 반전"""
        if self.input_format == 'NCHW':
            # Width 차원은 인덱스 3
            return x_batch[:, :, :, ::-1]
        elif self.input_format == 'NHWC':
            # Width 차원은 인덱스 2
            return x_batch[:, :, ::-1, :]