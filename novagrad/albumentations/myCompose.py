from ..backend import Backend


class MyCompose:
    """
    여러 증강 기법을 묶어서 실행하고, 백엔드 인스턴스를 하위 모듈에 전파합니다.
    """
    def __init__(self, transforms: list, backend: Backend):
        self.backend = backend
        self.transforms = []
        
        # 하위 변환 객체들에 동일한 backend 주입
        for t in transforms:
            t.backend = self.backend
            self.transforms.append(t)

    def __call__(self, x_batch):
        for transform in self.transforms:
            x_batch = transform(x_batch)
        return x_batch