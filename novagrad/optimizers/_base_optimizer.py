from ..backend import Backend

class BaseOptimizer:
    def __init__(
            self, learning_rate=0.001, backend: Backend = None, **kwargs
    ):
        self.backend = backend
        self.learning_rate = learning_rate

    def update(self, params, grads, *args):
        raise NotImplementedError("옵티마이저의 update() 메서드가 구현되지 않았습니다.")