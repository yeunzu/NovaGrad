from .base_optimizer import BaseOptimizer

class Momentum(BaseOptimizer):
    def __init__(self, learning_rate=0.001, backend = None, 
                 momentum=0.9, 
                 **kwargs):
        super().__init__(learning_rate, backend, **kwargs)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads, *args):
        if self.v is None:
            self.v = [self.backend.zeros_like(p) for p in params]

        new_params = []
        new_v = []

        for p, g, v in zip(params, grads, self,v):
            v_new = self.momentum * v - self.learning_rate * g
            new_v.append(v_new)
            new_params.append(p + v_new)

        self.v = new_v
        return new_params