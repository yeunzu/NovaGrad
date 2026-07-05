from .base_loss import BaseLoss

class MeanSquaredError(BaseLoss):
    def __init__(self, backend=None):
        super().__init__(backend)

    def forward(self, y, t, *args):
        batch_size = y.shape[0]
        # (예측값 - 정답)^2 의 평균
        return self.backend.sum((y - t) ** 2) / batch_size

    def backward(self, y, t, *args):
        batch_size = t.shape[0]
        # MSE의 미분: 2 * (y - t) / N
        dx = 2 * (y - t) / batch_size
        return dx