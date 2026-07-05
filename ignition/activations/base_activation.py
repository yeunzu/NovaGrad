from ..backend import Backend

class BaseActivation:
    def __init__(self, backend: Backend = None, **kwargs):
        self.backend = backend

    def forward(self, x, **kwargs):
        raise NotImplementedError("활성화함수의 forward 함수가 정의되지 않음")
    
    def backward(self, dout, **kwargs):
        raise NotImplementedError("활성화함수의 backward 함수가 정의되지 않음")
    
    def update(self):
        pass