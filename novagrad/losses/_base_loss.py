from ..backend import Backend

class BaseLoss:
    def __init__(self, backend: Backend = None):
        self.backend = backend

    def forward(self, y, t, *args):
        raise NotImplementedError("손실함수에 forward 함수가 정의되지 않음")
    
    def backward(self, y, t, *args):
        raise NotImplementedError("손실함수에 backward 함수가 정의되지 않음")