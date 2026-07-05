from ..backend import Backend
from ..optimizers.base_optimizer import BaseOptimizer
from ..optimizers.adam import Adam


class BaseLayer:
    def __init__(
            self, backend: Backend=None, 
            optimizer: BaseOptimizer = None, 
            learning_rate=0.001, 
            **opt_params
    ):
        self.backend = backend

        self.learning_rate = learning_rate
        # 가중치가 없는 레이어(Pooling 등)도 있으므로 optimizer는 옵션으로 둡니다.
        if optimizer is not None:
            self.optimizer = optimizer(learning_rate=self.learning_rate, backend=self.backend, **opt_params)
        else:
            self.optimizer = None

        # 모든 레이어는 params와 grads 리스트를 가지는 것이 관례입니다 (없으면 빈 리스트)
        self.params = []
        self.grads = []

    def forward(self, x, **kwargs):
        raise NotImplementedError("레이어에 forward 함수가 정의되지 않음")

    def backward(self, dout, **kwargs):
        raise NotImplementedError("레이어에 backward 함수가 정의되지 않음")
    
    def update(self):
        # 파라미터가 없으면 업데이트 생략
        if not self.params or self.optimizer is None:
            return
        self.params = self.optimizer.update(self.params, self.grads)
    
    def output_shape(self, input_shape):
        raise NotImplementedError("레이어에 output_shape 함수가 정의되지 않음")
    
    def astype(self, dtype_str):
        if hasattr(self, 'params') and self.params is not None:
            self.params = [self.backend.cast(p, dtype_str) for p in self.params]