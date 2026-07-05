from ..backend import Backend
from .base_layer import BaseLayer


class Dropout(BaseLayer):
    def __init__(
            self, backend = None, optimizer = None, learning_rate=0.001, 
            dropout_ratio=0.5, train_flg=True, 
            **opt_params):
        super().__init__(backend, optimizer, learning_rate, **opt_params)

        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = train_flg

    def forward(self, x, train_flg=None, **kwargs):
        if train_flg == None: train_flg = self.train_flg

        if train_flg:
            self.mask = self.backend.rand(x.shape)
            self.mask = self.mask > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout, **kwargs):
        return dout * self.mask
    
    # update는 부모 클래스 사용

    def output_shape(self, input_shape):
        return input_shape
    
    # astype은 부모 클래스 사용