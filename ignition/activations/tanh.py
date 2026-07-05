from .base_activation import BaseActivation

class Tanh(BaseActivation):
    def __init__(self, backend = None, **kwargs):
        super().__init__(backend, **kwargs)
        self.y = None

    def forward(self, x, **kwargs):
        self.y = (self.backend.exp(x) - self.backend.exp(-x)) / (self.backend.exp(x) + self.backend.exp(-x))
        return self.y
    
    def backward(self, dout, **kwargs):
        Ddout = dout * (1 - self.y**2)
        self.y = None
        return Ddout
    
    def update(self):
        return super().update()
    
    