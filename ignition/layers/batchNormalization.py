from ..backend import Backend
from .base_layer import BaseLayer

from ..optimizers.adam import Adam
from ..initializers.he import He


class BatchNormalization(BaseLayer):
    def __init__(self, backend: Backend = None, 
                 optimizer=Adam, learning_rate=0.001, 
                 momentum=0.9, running_mean=None, running_var=None, eps=1e-7, 
                 **opt_params):
        super().__init__(backend, optimizer, learning_rate, **opt_params)
        
        self.momentum = momentum
        self.eps = eps

        self.input_shape = None # Conv 레이어 대응용 (4차원)

        # gamma: scale 파라미터
        # delta: shift 파라미터
        self.params = [
            None, None
        ]
        self.grads = [
            None, None
        ]

        # 시험 때 이용할 평균, 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # 역전파 시 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.xn = None

        self.train_flg = True
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate=self.learning_rate, backend=self.backend, **opt_params)

    def forward(self, x, train_flg=None, **kwargs):
        if train_flg is None: train_flg = self.train_flg
        self.input_shape = x.shape

        if x.ndim != 2:
            N, C, H, W = x.shape
            x = self.backend.reshape(x, (N, -1))

        out = self.__forward(x, train_flg)

        return self.backend.reshape(out, self.input_shape)
    
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            D = x.shape[1]
            if self.running_mean is None:
                self.running_mean = self.backend.zeros(D) # 본래 넘파이로 되어 있어 백엔드로 변환
                self.running_var = self.backend.zeros(D)

                self.params[0] = self.backend.ones(D) # gamma
                self.params[1] = self.backend.zeros(D) # delta

                self.grads[0] = self.backend.zeros(D)
                self.grads[1] = self.backend.zeros(D)

        if train_flg:
            mu = self.backend.mean(x, axis=0) # 본래 x.mean(axis=0) 으로 되어 있던 것을 수정
            xc = x - mu
            var = self.backend.mean(xc**2, axis=0)
            std = self.backend.sqrt(var + self.eps)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        else:
            xc = x - self.running_mean
            xn = xc / ((self.backend.sqrt(self.running_var + self.eps)))

        out = self.params[0] * xn + self.params[1]

        return out
    
    def backward(self, dout, **kwargs):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx
    
    def __backward(self, dout):
        ddelta = self.backend.sum(dout, axis=0) # 본래 dout.sum(axis=0) 이었던 것을 수정
        dgamma = self.backend.sum(self.xn * dout, axis=0)
        dxn = self.params[0] * dout
        dxc = dxn / self.std
        dstd = -self.backend.sum((dxn * self.xc) / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc = dxc + ((2.0 / self.batch_size) * self.xc * dvar)
        dmu = self.backend.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.grads[0] = dgamma
        self.grads[1] = ddelta

        return dx
    
    # update는 부모 클래스 사용

    def astype(self, dtype_str):
        super().astype(dtype_str)
        if hasattr(self, 'running_mean') and self.running_mean is not None:
            self.running_mean = self.backend.cast(self.running_mean, dtype_str)
            self.running_var = self.backend.cast(self.running_var, dtype_str)

