from .base_activation import BaseActivation

PI = 3.141592653589793

class Gelu(BaseActivation):
    def __init__(self, backend=None, **kwargs):
        super().__init__(backend, **kwargs)
        self.x = None

    def forward(self, x, **kwargs):
        # 역전파 때 써야 하므로 원본 x를 저장해 둡니다.
        self.x = x
        
        # cdf = 0.5 * (1 + erf(x / sqrt(2)))
        # cdf = 0.5 * (1 + self.backend.erf(x / math.sqrt(2.0)))
        cdf = 0.5 * (1.0 + self.backend.erf(x / 2.0**0.5))
        
        out = x * cdf
        return out
    
    def backward(self, dout, **kwargs):
        # 역전파 수식: dout * (CDF + x * PDF)
        
        # 1. CDF 다시 계산 (또는 forward에서 self.cdf로 저장해두고 재활용해도 좋습니다)
        cdf = 0.5 * (1.0 + self.backend.erf(self.x / 2.0**0.5))
        
        # 2. PDF 계산: exp(-0.5 * x^2) / sqrt(2 * pi)
        pdf = self.backend.exp(-0.5 * (self.x ** 2)) / (2.0 * PI)**0.5
        
        # 3. 최종 기울기
        dx = dout * (cdf + self.x * pdf)
        return dx
    
    def update(self):
        return super().update()