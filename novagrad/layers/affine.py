from ..backend import Backend
from .base_layer import BaseLayer

from ..initializers.he import He
from ..optimizers.adam import Adam


class Affine(BaseLayer):
    def __init__(
            self, backend: Backend = None, 
            input_size=None, output_size=None, 
            initializer=He, optimizer = Adam, 
            learning_rate=0.001, **opt_params):
        super().__init__(backend, optimizer, learning_rate, **opt_params)

        self.params = [
            initializer(input_size, output_size, backend), 
            self.backend.zeros(output_size)
        ]
        self.grads = [
            self.backend.zeros_like(self.params[0]), 
            self.backend.zeros_like(self.params[1])
        ]

        # He, Xavier 등 구체적인 초기화는 파라미터가 필요한 자식 클래스가 직접 수행
        init_fn = initializer() # 초기화 클래스 인스턴스화
        W = init_fn(input_size, output_size, backend)
        B = self.backend.zeros(output_size)
        
        self.params = [W, B]
        self.grads = [self.backend.zeros_like(W), self.backend.zeros_like(B)]

        self.x = None

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x, **kwargs):
        self.x = x
        return self.backend.matmul(x, self.params[0]) + self.params[1]
    
    def backward(self, dout, **kwargs):
        dx = self.backend.matmul(dout, self.params[0].T)
        dW = self.backend.matmul(self.x.T, dout)
        self.x = None # 메모리 절약
        dB = self.backend.sum(dout, axis=0)
        self.grads[0] = dW
        self.grads[1] = dB
        return dx
    
    # 부모 클래스 사용
    # def update(self):
    #     self.params = self.optimizer.update(self.params, self.grads)

    def output_shape(self, input_shape):
        return (self.output_shape, )
    
    # 부모 클래스 사용
    # def astype(self, dtype_str):
    #     return super().astype(dtype_str)
    