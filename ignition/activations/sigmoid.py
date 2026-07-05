from .base_activation import BaseActivation

class Sigmoid(BaseActivation):
    def __init__(self, backend = None, **kwargs):
        super().__init__(backend, **kwargs)
        self.y = None

    def forward(self, x, **kwargs):
        self.y = 1 / (1 + self.backend.exp(-x))
        return self.y
    
    def backward(self, dout, **kwargs):
        Ddout = dout * (self.y * (1 - self.y))
        self.y = None
        return Ddout
    
    def update(self):
        return super().update()
    