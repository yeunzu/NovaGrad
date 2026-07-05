from .base_activation import BaseActivation

class Softmax(BaseActivation):
    def __init__(self, backend = None, **kwargs):
        super().__init__(backend, **kwargs)
        self.y = None

    def forward(self, x, **kwargs):
        c = self.backend.max(x, axis=1, keepdims=True)
        exp_a = self.backend.exp(x - c)
        sum_exp_a = self.backend.sum(exp_a, axis=1, keepdims=True)
        self.y = exp_a / sum_exp_a
        return self.y
    
    def backward(self, dout, **kwargs):
        Ddout = self.y * (dout - self.backend.sum(dout * self.y, axis=1, keepdims=True))
        self.y = None
        return Ddout
    
    def update(self):
        return super().update()