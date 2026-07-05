from ..backend import Backend
from ..models.sequential import MyModel
from ..dataloaders.dataloader import DataLoader

class Trainer:
    """
    MyModel + DataLoader 를 받아 학습 루프 전체를 담당
    
    Parameters
    ---------
    model           : MyModel 인스턴스
    train_loader    : DataLoader - 학습용 (필수)
    val_loader      : DataLoader - 검증용 (선택)
    test_loader     : DataLoader - 테스트용 (선택, 보통 fit 종료 후 1회)
    backend         : Backend - 하드웨어 가속 의존성 주입 (필수)
    """

    def __init__(self, model: MyModel, train_loader: DataLoader, val_loader: DataLoader =None, test_loader: DataLoader =None, backend: Backend =None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


        # [수정] 모델이 가지고 있는 백엔드를 최우선으로 공유받음
        self.backend = backend if backend is not None else getattr(model, 'backend', None)

        
        # 기록 저장소
        # key: 'train' | 'val' | 'test'
        # value : {'epoch':[], 'loss':[], 'acc':[]}
        self._log = {
            'train' : {'epoch': [], 'loss': [], 'acc': []}, 
            'val' : {'epoch': [], 'loss': [], 'acc': []}, 
            'test' : {'epoch': [], 'loss': [], 'acc': []}
        }


    # # 한 에포크 순전파 (공용)
    # def _run_epoch(self, loader, train: bool):
    #     """
    #     loader를 한 바퀴 돌며 손실저정확도 계산
    #     train=True -> 역전파+가중치 갱신 수행
    #         =False ->수순전파만 수행 (val / test용)
    #     """
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0

    #     for x_batch, t_batch in loader:
    #         y, loss = self.model.get_loss(x_batch, t_batch)

    #         if train:
    #             # 여기선 jit 컴파일 X. 컴파일 시 외부 변수 수정이 안되어 에러 발생 가능
    #             self.model.backward(y, t_batch)
    #             self.model.update()

    #         total_loss += float(loss)
    #         # 정확도: one-hot 또는 class index 레이블 모두 지원
    #         pred = np.argmax(np.array(y), axis=1)
    #         if t_batch.ndim == 2: # one-hot
    #             true = np.argmax(np.array(t_batch), axis=1)
    #         else:
    #             true = np.array(t_batch)
    #         correct += int(np.sum(pred == true))
    #         total += t_batch.shape[0]

    #     avg_loss = total_loss / len(loader)
    #     avg_acc = correct / total
    #     return avg_loss, avg_acc

    def _run_epoch(self, loader, train: bool):
        """
        loader를 한 바퀴 돌며 손실저정확도 계산
        train=True -> 역전파+가중치 갱신 수행
             =False ->순전파만 수행 (val / test용)
        """
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, t_batch in loader:
            y, loss = self.model.get_loss(x_batch, t_batch, train_flg=train)

            if train:
                # 여기선 jit 컴파일 X. 컴파일 시 외부 변수 수정이 안되어 에러 발생 가능
                self.model.backward(y, t_batch)
                self.model.update()

            # [수정] 가속기 상에서 텐서끼리 덧셈
            total_loss = total_loss + loss

            # [수정] 강제 CPU 변환 제거 -> 백엔드 네이티브 연산으로 정확도 계산
            pred = self.backend.argmax(y, axis=1)
            
            if t_batch.ndim == 2: # one-hot
                true = self.backend.argmax(t_batch, axis=1)
            else:   
                true = t_batch
                
            # [수정] 가속기에서 일치 여부(==)와 합(sum)을 끝낸 뒤, 마지막 결과 1개만 가져옴
            matched = self.backend.sum(pred == true)
            correct = correct + matched
            total += t_batch.shape[0]

        # [핵심 주의] for문(1에포크)이 완전히 끝난 뒤, 딱 한 번만 CPU로 값을 가져옴 (동기화 1회 발생)
        total_loss = self.backend.cast(total_loss, 'fp32') # dtype 캐스팅으로 넘파이에서 지원 안될 경우를 방지하기 위한 cast
        correct = self.backend.cast(correct, 'fp32')
        avg_loss = float(self.backend.to_numpy(total_loss)) / len(loader)
        avg_acc = float(self.backend.to_numpy(correct)) / total
        return avg_loss, avg_acc

    # 기록 저장용
    def _record(self, split: str, epoch: int, loss: float, acc: float):
        self._log[split]['epoch'].append(epoch)
        self._log[split]['loss'].append(loss)
        self._log[split]['acc'].append(acc)

    # 학습
    def fit(
            self, 
            epochs: int,
            checkpoint_every: int = None,
            checkpoint_path: str = 'chechpoint.npz',
            eval_test_every: int = None, 
            verbose: bool = True
    ): 
        """
        학습 실행

        Parameters
        ---------
        epochs              : 총 에포크 수
        chechpoint_every    : N 에포크마다 모델을 pkl로 저장. None이면 저장 안 함.
        checkpoint_path     : 저장 경로 템플릿. '_epochN'이 자동으로 삽입됨
                            ex] 'chpt.npz' -> 'chpt_epoch5.npz'
        eval_test_every     : N 에포크마다 test 데이터 평가. None이면 fit 종료 후 1회만.
        verbose             : 에포크별 출력 여부
        """
        for epoch in range(1, epochs + 1):
            # train
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            self._record('train', epoch, tr_loss, tr_acc)

            # val
            if self.val_loader is not None:
                val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
                self._record('val', epoch, val_loss, val_acc)

            # test (매 N 에포크)
            if (self.test_loader is not None) and (eval_test_every is not None):
                if (epoch % eval_test_every == 0):
                    te_loss, te_acc = self._run_epoch(self.test_loader, train=False)
                    self._record('test', epoch, te_loss, te_acc)

            # 출력
            if verbose:
                self._print_epoch(epoch, epochs)

            # # checkpoint
            # if checkpoint_every and (epoch % checkpoint_every == 0):
            #     path = checkpoint_path.replace('.pkl', f'_epoch{epoch}.pkl')
            #     self.model.save(path)

            # (fit 메서드 내부의 checkpoint 부분 수정)
            # checkpoint
            if checkpoint_every and (epoch % checkpoint_every == 0):
                # .pkl 대신 .npz로 파일 확장자 변경
                path = checkpoint_path.replace('.npz', f'_epoch{epoch}.npz')
                # 모델 전체 저장이 아닌 가중치 배열 추출 저장
                self.model.save_weights(path)

        # fit 종료 후 test 1회 평가
        # eval_test_every 미지정 시에만 작동
        if (self.test_loader is not None) and (eval_test_every is None):
            te_loss, te_acc = self._run_epoch(self.test_loader, train=False)
            self._record('test', epochs, te_loss, te_acc)
            if verbose:
                print(f"    [test] loss: {te_loss:.4f} | acc: {te_acc:.4f}")

    def _print_epoch(self, epoch: int, total: int):
        """
        에포크 진행 상황 출력
        """
        width = len(str(total))
        line  = f"Epoch {epoch:{width}d}/{total}"
 
        tr = self._log['train']
        line += (f"  │  train  loss: {tr['loss'][-1]:.4f}"
                 f"  acc: {tr['acc'][-1]:.4f}")
 
        if self._log['val']['epoch']:
            v = self._log['val']
            line += (f"  │  val    loss: {v['loss'][-1]:.4f}"
                     f"  acc: {v['acc'][-1]:.4f}")
 
        # test는 매 에포크 찍히지 않을 수 있으므로 현재 에포크에 기록된 경우만
        te = self._log['test']
        if te['epoch'] and te['epoch'][-1] == epoch:
            line += (f"  │  test   loss: {te['loss'][-1]:.4f}"
                     f"  acc: {te['acc'][-1]:.4f}")
 
        print(line)

    # 기록 변환 메서드
    def history(self) -> dict:
        """
        모든 기록을 하나의 딕셔너리로 반환.

        Returns
        -------
        {
            'epoch'      : [1, 2, 3, ...],   # train 기준
            'train_loss' : [...],
            'train_acc'  : [...],
            'val_loss'   : [...],   # val_loader 없으면 빈 리스트
            'val_acc'    : [...],
            'val_epoch'  : [...],   # val 기록 에포크 (train과 동일하면 생략 가능)
            'test_loss'  : [...],
            'test_acc'   : [...],
            'test_epoch' : [...],   # test 평가가 매 에포크가 아닐 수 있으므로 별도 제공
        }
        """
        return {
            'epoch':      self._log['train']['epoch'].copy(),
            'train_loss': self._log['train']['loss'].copy(),
            'train_acc':  self._log['train']['acc'].copy(),
            'val_epoch':  self._log['val']['epoch'].copy(),
            'val_loss':   self._log['val']['loss'].copy(),
            'val_acc':    self._log['val']['acc'].copy(),
            'test_epoch': self._log['test']['epoch'].copy(),
            'test_loss':  self._log['test']['loss'].copy(),
            'test_acc':   self._log['test']['acc'].copy(),
        }

    def train_history(self) -> tuple:
        """
        (epochs, losses, accs) 튜플로 반환
        """
        t = self._log['train']
        return t['epoch'].copy(), t['loss'].copy(), t['acc'].copy()
    
    def val_history(self) -> tuple:
        """(epochs, losses, accs) - val_loader 없으면 빈 리스트 3개"""
        v = self._log['val']
        return v['epoch'].copy(), v['loss'].copy(), v['acc'].copy()
    
    def test_history(self) -> tuple:
        """
        (epochs, losses, accs) - eval_test_every 미우정 시 마지막 에포크 기록 하나만
        """
        t = self._log['test']
        return t['epoch'].copy(), t['loss'].copy(), t['acc'].copy()

    def best_val_epoch(self) -> dict:
        """
        val_loss가 가장 낮았던 에포크와 그때의 지표를 반환

        Returns
        ------
        { 'epoch': int, 'val_loss': float, 'val_acc': float }
        또는 val 기록이 없으면 None
        """
        v = self._log['val']
        if not v['loss']:
            return None
        best_idx = int(self.backend.argmin(v['loss'])) # 본래 int(np.argmin(v['loss']))였던 걸 변환
        return {
            'epoch':    v['epoch'][best_idx],
            'val_loss': v['loss'][best_idx],
            'val_acc':  v['acc'][best_idx],
        }