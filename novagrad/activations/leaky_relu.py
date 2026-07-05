from .base_activation import BaseActivation

class LeakyRelu(BaseActivation):
    def __init__(self, backend = None, alpha = 0.01, **kwargs):
        super().__init__(backend, **kwargs)
        self.mask_pos = None
        self.mask_neg = None
        self.alpha = alpha

    def forward(self, x, **kwargs):
        self.mask = (x > 0)
        out = self.backend.where(self.mask, x, x*self.alpha)
        return out
    
    def backward(self, dout, **kwargs):
        dx = self.backend.where(self.mask, dout, dout*self.alpha)
        return dx
    
    def update(self):
        return super().update()