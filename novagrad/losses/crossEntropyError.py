from .base_loss import BaseLoss

class CrossEntropyError(BaseLoss):
    def __init__(self, backend = None, eps=1e-8):
        super().__init__(backend)
        self.eps = eps

    def forward(self, y, t, *args):
        return -self.backend.sum(t * self.backend.log(y + self.eps)) / y.shape[0]

    def backward(self, y, t, *args):
        batch_size = t.shape[0]
        return (-t / (y + self.eps)) / batch_size