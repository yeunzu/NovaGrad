from ..backend import Backend


class BaseTransform:
    """
    모든 커스텀 증강 클래스의 부모 클래스.
    확률 제어 및 배치 단위 마스킹(Vectorization) 로직을 담당합니다.
    """
    def __init__(self, p: float = 0.5):
        self.p = p
        self.backend : Backend = None  # Compose에서 주입됨

    def __call__(self, x_batch):
        if self.p == 0:
            return x_batch
        if self.p == 1:
            return self.apply(x_batch)

        # 1. 배치 크기 추출
        N = x_batch.shape[0] # (N, C, H, W)의 첫 번째

        # 2. 브로드캐스팅을 위한 마스크 형태 생성 (N, 1, 1, 1)
        # 4D 텐서(NCHW 또는 NHWC) 기준
        mask_shape = (N,) + (1,) * (len(x_batch.shape) - 1)

        # 3. 0~1 사이 균등분포 난수 생성 및 확률 비교
        # 각 이미지마다 독립적으로 적용 여부(True/False)가 결정됨
        rands = self.backend.rand(mask_shape)
        condition = rands < self.p

        # 4. 전체 배치에 대해 변환 로직 실행 (JIT 컴파일 시 병렬 처리됨)
        transformed_x = self.apply(x_batch)

        # 5. 마스크에 따라 원본과 변환된 이미지를 취사선택 (Vectorized)
        return self.backend.where(condition, transformed_x, x_batch)

    def apply(self, x_batch):
        """실제 수학적/배열 변환 로직. 하위 클래스에서 반드시 구현해야 함."""
        raise NotImplementedError