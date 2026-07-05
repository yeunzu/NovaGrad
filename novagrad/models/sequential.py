from collections import OrderedDict
import pickle

from ..backend import Backend

class MyModel:
    def __init__(self, backend: Backend):
        self.backend = backend # 백엔드 주입

        self._layers: OrderedDict = OrderedDict()
        self._loss_func = None

        # 자동 이름 부여용 카운터
        self._name_counter: dict = {}

    def _auto_name(self, layer) -> str:
        """이름을 지정하지 않으면 'Affine_1', 'Relu_1' 형태로 자동 부여"""
        cls =type(layer).__name__
        count = self._name_counter.get(cls, 0) + 1
        self._name_counter[cls] = count
        return f"{cls}_{count}"

    def add_layer(self, layer, name: str = None, index: int = None):
        """
        레이어 추가

        Parameters
        ---------
        layer   : 레이어 인스턴스
        name    : 레이어 이름. 없으면 자동 부여
        index   : 삽입 위치. None임면 맨 끝에 추가. 0이면 맨 앞, -1이면 맨 끝 앞
        """
        if name is None:
            name = self._auto_name(layer)
        if name in self._layers:
            raise ValueError(f"이미 '{name}' 이름의 레이어가 존재합니다.")
        
        if index is None:
            self._layers[name] = layer
        else:
            # OrderedDict는 중간 삽입을 직접 지원하지 않으므로 재구성
            items = list(self._layers.items())
            if index < 0:
                index = len(items) + 1 + index
            items.insert(index, (name, layer))
            self._layers = OrderedDict(items)

    def remove_layer(self, key):
        """
        이름(str) 또는 인덱스(int)로 레이어를 제거.
        """

        name = self._resolve_key(key)
        del self._layers[name]

    def get_layer(self, key):
        """이름(str) 또는 인덱스(int)로 레이어를 반환"""
        return self._layers[self._resolve_key(key)]
    
    def __getitem__(self, key):
        """model['conv1'] 또는 model[0] 으로 레이어 접근"""
        return self.get_layer(key)

    def _resolve_key(self, key) -> str:
        """str/int 키를 레이어 이름(str)으로 변환"""
        if isinstance(key, str):
            if key not in self._layers:
                raise KeyError(f"'{key}' 이름의 레이어가 없습니다.")
            return key
        elif isinstance(key, int):
            names = list(self._layers.keys())
            if (key < -len(names)) or (key >= len(names)):
                raise IndexError(f"인덱스 {key}가 범위를 벗어났습니다. (레이어 수 : {len(names)})")
            return names[key]
        else:
            raise TypeError(f"키는 str 또는 int 여야 합니다. (받은 타입: {type(key).__name__})")
        
    @property
    def layers(self) -> list:
        """레이어 인스턴스 리스트 (순서 보장)"""
        return list(self._layers.values())
    
    @property
    def layer_names(self) -> list:
        return list(self._layers.keys())

    def set_loss_func(self, func):
        """
        손실함수 설정
        loss_func은 forward(y, t) -> scalar 를 구현해야 함
        """
        self._loss_func = func

    def astype(self, dtype_str):
        print(f"모델의 모든 가중치를 {dtype_str}로 변환")
        for layer in self.layers:
            if hasattr(layer, 'astype'):
                layer.astype(dtype_str)
    
    # 순전파, 손실, 역전파, 갱신
    def predict(self, x, **kwargs):
        for layer in self.layers:
            x = layer.forward(x, **kwargs)
        return x
    
    def get_loss(self, x, t, **kwargs):
        if self._loss_func is None:
            raise RuntimeError("손실 함수가 설정되지 않았습니다. set_loss_func()을 먼저 호출하세요.")

        y = self.predict(x, **kwargs)
        loss = self._loss_func.forward(y, t)
        return y, loss

    def backward(self, y, t):
        dout = self._loss_func.backward(y, t)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update(self):
        for layer in self.layers:
            layer.update()

    # 저장 및 불러오기
    # def save(self, path: str):
    #     """
    #     모델 전체(구조 + 가중치 + 옵아마이저 상태)를 pickle로 저장.
    #     나중에 MyModel.load(path)로 완전히 복원 가능
    #     """
    #     with open(path, 'wb') as f:
    #         pickle.dump(self, f)
    #     print(f"[MyModel] 모델 저장 완료 -> {path}")

    # @staticmethod
    # def load(path: str) -> 'MyModel':
    #     """
    #     pickle로 저장된레모델을 불러옴
    #     """
    #     with open(path, 'rb') as f:
    #         model = pickle.load(f)
    #     print(f"[MyModel] 모델 불러오기 완료 <- {path}")
    #     return model

    def save_weights(self, path: str):
        """
        가중치만 numpy .npz 형식으로 저장
        모델 구조는 저장되지 않으므로, 불러올 때는 동일한 구조의 모델을 먼저 만들고 load_weights()를 호출하애 함.
        """
        # 내부 레이어를 재귀적으로 파고들며 가중치를 캐내는 내부 함수
        def _extract_state(layer):
            state = {}
            # 1. 단일 레이어의 파라미터 저장
            if hasattr(layer, 'params') and layer.params:
                state['params'] = [self.backend.to_numpy(p) for p in layer.params]
            if hasattr(layer, 'running_mean') and layer.running_mean is not None:
                state['running_mean'] = self.backend.to_numpy(layer.running_mean)
                state['running_var'] = self.backend.to_numpy(layer.running_var)
                
            # 2. ResidualBlock 같은 복합 레이어의 내부 파고들기!
            if hasattr(layer, 'internal_layers') and layer.internal_layers:
                state['internal_layers'] = [_extract_state(inner) for inner in layer.internal_layers]
                
            return state

        # 전체 레이어 순회
        full_state = [_extract_state(layer) for layer in self.layers]

        with open(path, 'wb') as f:
            pickle.dump(full_state, f)
        print(f"💾 [MyModel] 모델 가중치(중첩 블록 포함) 안전 저장 완료 -> {path}")

    def load_weights(self, path="model_weights.pkl"):
        import pickle
        
        # 저장된 딕셔너리를 보고 재귀적으로 가중치를 이식하는 내부 함수
        def _apply_state(layer, state):
            # 1. 단일 레이어 파라미터 복원 (현재 fp32 기준)
            if 'params' in state:
                layer.params = [self.backend.cast(self.backend.array(p), 'fp32') for p in state['params']]
                if hasattr(layer, 'W'):
                    layer.W, layer.B = layer.params
            if 'running_mean' in state:
                layer.running_mean = self.backend.cast(self.backend.array(state['running_mean']), 'fp32')
                layer.running_var = self.backend.cast(self.backend.array(state['running_var']), 'fp32')
                
            # 2. 복합 레이어(ResidualBlock) 내부 이식!
            if 'internal_layers' in state and hasattr(layer, 'internal_layers'):
                for inner_layer, inner_state in zip(layer.internal_layers, state['internal_layers']):
                    _apply_state(inner_layer, inner_state)

        with open(path, 'rb') as f:
            full_state = pickle.load(f)

        for layer, state in zip(self.layers, full_state):
            _apply_state(layer, state)

        print(f"🔄 [MyModel] 모델 가중치(중첩 블록 포함) 성공적으로 복원 완료 -> {path}")

    # 요약 출력
    def summary(self, input_shape: tuple = None):
        """
        모델 구조 출력

        Parameters
        ---------
        input_shape : 배치 제외한 입력 shape. ex] (1, 28, 28) 또는 (784,)
                    None이면 outpht shape 열은 '?'로 표시
        """
        COL = {'name': 16, 'type': 15, 'output': 14, 'params': 10}
        SEP  = '─'
        CORN = ('┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '│')
        W = sum(COL.values()) + len(COL) * 3 - 1  # 전체 너비
 
        def hline(l, m, r, s='─'):
            parts = [s * (v + 2) for v in COL.values()]
            return l + s.join(parts) + r  # ← 수정: s.join 사용
 
        def row(name, typ, out, params):
            return (f"│ {name:<{COL['name']}}│ {typ:<{COL['type']}}│"
                    f" {out:<{COL['output']}}│ {params:>{COL['params']}} │")
 
        print(hline('┌', '┬', '┐'))
        title = 'Model Summary'
        print(f"│ {title:<{W}} │")
        print(hline('├', '┼', '┤'))
        print(row('Name', 'Type', 'Output', 'Params'))
        print(hline('├', '┼', '┤'))
 
        total_params = 0
        current_shape = input_shape  # None이면 shape 추적 안 함
 
        for name, layer in self._layers.items():
            cls_name = type(layer).__name__
 
            # 파라미터 수 계산
            params_list = getattr(layer, 'params', [])
            n_params = sum(p.size for p in params_list)
            total_params += n_params
 
            # 출력 shape 추적
            if current_shape is not None and hasattr(layer, 'output_shape'):
                try:
                    current_shape = layer.output_shape(current_shape)
                    out_str = str(current_shape)
                except Exception:
                    out_str = '?'
            elif current_shape is not None:
                # output_shape 미구현 레이어는 shape 추적 포기
                out_str = '?'
                current_shape = None
            else:
                out_str = '?'
 
            param_str = f"{n_params:,}" if n_params > 0 else '0'
            print(row(name[:COL['name']], cls_name[:COL['type']], out_str[:COL['output']], param_str))
 
        print(hline('├', '┼', '┤'))
        # 총 파라미터 수 행
        total_label = 'Total parameter'
        total_str = f"{total_params:,}"
        pad = COL['name'] + COL['type'] + COL['output'] + 6  # 열 3개 + 구분자
        print(f"│ {total_label:<{pad}}│ {total_str:>{COL['params']}} │")
        print(hline('└', '┴', '┘'))