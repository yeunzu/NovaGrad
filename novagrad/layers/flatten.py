from ..backend import Backend
from .base_layer import BaseLayer

class Flatten(BaseLayer):
    def __init__(self, backend: Backend = None, **opt_params):
        self.backend = backend
        self.original_x_shape = None
        self.params = []
        self.grads = []

    def forward(self, x, **kwargs):
        self.original_x_shape = x.shape
        if x.ndim > 2:
            return self.backend.reshape(x, (x.shape[0], -1))
        else:
            return self.backend.reshape(x, (1, -1))
        
    def backward(self, dout, **kwargs):
        return self.backend.reshape(dout, self.original_x_shape)
    
    # update는 부모 클래스 사용
    
    def output_shape(self, input_shape):
        return (int(self.backend.prod(input_shape)), )
    
    # astype은 부모 클래스 사용