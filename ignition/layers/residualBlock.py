from ..backend import Backend
from .base_layer import BaseLayer

from ..optimizers.adam import Adam
from ..initializers.he import He

from .conv import Conv
from ..activations.relu import Relu

class ResidualBlock(BaseLayer):
    def __init__(self, 
                 filter_num, filter_height, filter_width, stride=1, pad=1, 
                 backend: Backend = None, initializer=He, optimizer = Adam, learning_rate=0.001, **opt_params):
        super().__init__(backend, optimizer, learning_rate, **opt_params)

        self.conv1 = Conv(
            filter_num=filter_num, filter_channels=filter_num, filter_height=filter_height, filter_width=filter_width, 
            stride=stride, pad=pad, initializer=initializer, optimizer=optimizer, learning_rate=learning_rate, backend=self.backend, **opt_params
        )
        self.relu1 = Relu(backend=self.backend)
        self.conv2 = Conv(
            filter_num=filter_num, filter_channels=filter_num, filter_height=filter_height, filter_width=filter_width, 
            stride=stride, pad=pad, initializer=initializer, optimizer=optimizer, learning_rate=learning_rate, backend=self.backend, **opt_params
        )
        self.relu2 = Relu(backend=self.backend)
        self.internal_layers = [self.conv1, self.conv2]

    def forward(self, x, **kwargs):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = out + x # 마지막 Relu는 skip-connection 이후에 적용
        out = self.relu2.forward(out)

        return out + x
    
    def backward(self, dout, **kwargs):
        # 1. 가장 외곽의 최종 ReLU 역전파를 먼저 수행
        dact = self.relu2.backward(dout)
        
        # 2. Main Path 역전파
        dx_main = self.conv2.backward(dact)
        dx_main = self.relu1.backward(dx_main)
        dx_main = self.conv1.backward(dx_main)
        
        # 3. Skip Connection의 기울기(dact)를 주 경로의 기울기(dx_main)와 합산
        return dx_main + dact
    
    def update(self):
        for layer in self.internal_layers:
            layer.update()

    def output_shape(self, input_shape):
        tmp = self.conv1.output_shape(input_shape)
        return self.conv2.output_shape(tmp)

    def astype(self, dtype_str):
        for layer in self.internal_layers:
            layer.astype(dtype_str)