from ._base_optimizer import BaseOptimizer

class SGD(BaseOptimizer):
    def __init__(self, learning_rate=0.001, backend = None, **kwargs):
        super().__init__(learning_rate, backend, **kwargs)

    def update(self, params, grads, *args):
        return [p - self.learning_rate * g for p, g in zip(params, grads)]