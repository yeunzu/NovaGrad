from .base_loss import BaseLoss

class BinaryCrossEntropy(BaseLoss):
    def __init__(self, backend=None, eps=1e-8):
        super().__init__(backend)
        self.eps = eps

    def forward(self, y, t, *args):
        batch_size = y.shape[0]
        # BCE 수식: - [ t*log(y) + (1-t)*log(1-y) ]
        loss_pos = t * self.backend.log(y + self.eps)
        loss_neg = (1 - t) * self.backend.log(1 - y + self.eps)
        loss = -self.backend.sum(loss_pos + loss_neg)
        return loss / batch_size

    def backward(self, y, t, *args):
        batch_size = t.shape[0]
        # BCE 미분: - (t/y) + (1-t)/(1-y)
        # 0으로 나누어지는 것을 방지하기 위해 분모에 eps를 더해줍니다.
        dx_pos = - (t / (y + self.eps))
        dx_neg = (1 - t) / (1 - y + self.eps)
        dx = (dx_pos + dx_neg) / batch_size
        return dx