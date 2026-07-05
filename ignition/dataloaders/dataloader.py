from ..backend import Backend
# from ..albumentations import *

import numpy as np

class DataLoader:
    """
    Parameters
    ---------------
    x           : 배열 형태의 데이터 (np.ndarray, h5py.Dataset 등)
    y           : 배열 형태의 데이터 (np.ndarray, h5py.Dataset 등)
    batch_size  : int
    backend     : Backend           - 백엔드 인스턴스
    shuffle     : bool              - 에포크마다 인덱스 셔플 여부
    x_dtype     : str | None        - 'fp32', 'fp16', 'bf16', 'int32', 'int64'
    y_dtype     : str | None        - x_dtype과 동일
    drop_last   : bool              - 마지막 불완전 배치 버릴지 여부
    augmentor   : MyCompose | None  - 하드웨어 가속 데이터 증강 파이프라인
    """
    def __init__(self,
                 x, 
                 y, 
                 batch_size: int = 1,
                 backend: Backend = None, 
                 shuffle: bool = False,
                 x_dtype: str = None,
                 y_dtype: str = None,
                 drop_last: bool = False, 
                 augmentor = None # MyCompose 객체가 들어옴
    ):
        assert x.shape[0] == y.shape[0], 'x와 y의 샘플 수가 다릅니다.'
        assert batch_size > 0, 'batch_size는 1 이상이어야 합니다.'

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.backend = backend if backend is not None else Backend()
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.drop_last = drop_last
        self.augmentor = augmentor

        # h5py 데이터셋인지 확인하기 위한 플래그
        self.is_h5 = hasattr(x, 'chunks') # h5py 데이터셋은 chunks 속성을 가짐

        self.n = x.shape[0]
        self._num_batches = self.n // batch_size
        if not drop_last and (self.n % batch_size != 0):
            self._num_batches += 1

    def __len__(self) -> int:
        """총 배치 수"""
        return self._num_batches
    
    def __iter__(self):
        """
        에포크마다 새로 호출.
        HDF5 I/O 방어를 위해 청크(Chunk) 단위 셔플링 수행
        """
        # 연속된 배치 묶음(chunk)의 시작 인덱스들
        batch_starts = np.arange(0, self.n, self.batch_size)

        if self.shuffle:
            # 전체 데이터가 아닌, 배치를 가져오는 순서만 섞기
            # I/O 병목을 막기 위한 타협
            np.random.shuffle(batch_starts)

        for start in batch_starts:
            end = start + self.batch_size

            if self.drop_last and end>self.n:
                continue

            # 디스크(또는 메모리리에서 연속된 슬라이스 형태로 데이터를 불러움
            # 주의: h5py 배열에 끝 인덱스가 범위를 넘어서도 파이썬 슬라이싱처럼 잘라줌
            x_batch = self.x[start:end]
            y_batch = self.y[start:end]

            # numpy 배열이 아닌 경우 변환(h5py 데이터셋은 읽어올 때 넘파이 배열이 됨)
            if not isinstance(x_batch, np.ndarray):
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)


            # 증강 적용 (dtype 변환 전에 - uint8 변환은 Augmentor 내부에서 처리)
            if self.augmentor is not None:
                x_batch = self.augmentor(x_batch)

            # Numpy 배열 -> 가속기로 변환
            if type(x_batch).__module__ == 'numpy':
                x_batch = self.backend.array(x_batch)
            if type(y_batch).__module__ == 'numpy':
                y_batch = self.backend.array(y_batch)

            # dtype 변환 (지정된 경우에만)
            if self.x_dtype:
                x_batch = self.backend.cast(x_batch, self.x_dtype)
                
            if self.y_dtype:
                y_batch = self.backend.cast(y_batch, self.y_dtype)

            yield x_batch, y_batch

    def subset(self, ratio: float) -> 'DataLoader':
        """
        전체 데이터의 일부만 쓰는 DataLoader를 반환.
        빠른 디버깅 / 오버피팅 테스트용
        loader.subset(0.1)  ->  10%만 사용
        """
        assert 0 < ratio <= 1.0
        n_sub = max(1, int(self.n * ratio))
        
        return DataLoader(
            x=self.x[:n_sub], 
            y=self.y[:n_sub], 
            batch_size=self.batch_size,
            backend=self.backend, 
            shuffle=self.shuffle, 
            x_dtype=self.x_dtype, 
            y_dtype=self.y_dtype, 
            drop_last=self.drop_last
        )

