from .base_optimizer import BaseOptimizer

class Adam(BaseOptimizer):
    def __init__(self, learning_rate=0.001, backend = None, 
                 beta1=0.9, beta2=0.999, eps=1e-8, 
                 **kwargs):
        super().__init__(learning_rate, backend, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads, *args):
        if self.m is None:
            self.m = [self.backend.zeros_like(p) for p in params]
            self.v = [self.backend.zeros_like(p) for p in params]

        self.t += 1

        lr_t = self.lr * ((1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t))

        new_params = []
        new_m = []
        new_v = []

        for p, g, m, v in zip(params, grads, self.m, self.v):
            m_new = self.beta1 * m + (1 - self.beta1) * g
            v_new = self.beta2 * v + (1 - self.beta2) * (g * g)
            new_m.append(m_new)
            new_v.append(v_new)
            new_params.append(p - lr_t * m_new / (self.backend.sqrt(v_new) + self.eps))

        self.m = new_m
        self.v = new_v
        return new_params