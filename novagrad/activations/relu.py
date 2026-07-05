from .base_activation import BaseActivation

class Relu(BaseActivation):
    def __init__(self, backend = None, **kwargs):
        super().__init__(backend, **kwargs)
        self.mask = None

    def forward(self, x, **kwargs):
        self.mask = (x > 0)
        out = self.backend.where(self.mask, x, 0.0)
        return out
    
    def backward(self, dout, **kwargs):
        new_dout = self.backend.where(self.mask, dout, 0.0)
        return new_dout
    
    def update(self):
        return super().update()