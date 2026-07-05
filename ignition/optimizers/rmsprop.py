from .base_optimizer import BaseOptimizer

class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate=0.001, backend = None, 
                 rho=0.99, eps=1e-8, 
                 **kwargs):
        super().__init__(learning_rate, backend, **kwargs)
        
        self.rho = rho
        self.eps = eps

        self.h = None

    def update(self, params, grads, *args):
        if self.h is None:
            self.h = [self.backend.zeros_like(p) for p in params]

        new_params = []
        new_h = []
        for p, g, h in zip(params, grads, self.h):
            h_new = self.rho * h + (1 - self.rho) * g * g
            new_h.append(h_new)
            new_params.append(p - self.lr / (self.backend.sqrt(h_new) + self.eps) * g)

        self.h = new_h
        return new_params