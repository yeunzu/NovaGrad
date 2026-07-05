# 🔥 NovaGrad

[![PyPI version](https://badge.fury.io/py/novagrad.svg)](https://badge.fury.io/py/novagrad)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"인공지능의 혁신을 일으켜 전 지구적인 문제를 해결하는 점화 플러그"**

\
\
**NovaGrad**는 JAX와 MLX 백엔드를 완벽하게 추상화하여, 클라우드(Colab TPU/GPU)와 로컬(Mac 통합 메모리) 환경을 자유롭게 넘나드는 하이브리드 딥러닝 프레임워크입니다. 



## 🚀 왜 NovaGrad인가요?

1. **Hardware Agnostic:** 단 한 줄의 코드 수정 없이, 로컬 Mac(MLX)의 통합 메모리 이점과 클라우드(JAX)의 분산 처리 이점을 모두 누리세요.
2. **Pure & Transparent:** 밑바닥부터 직접 구현한 객체지향적 구조로, 블랙박스 없이 신경망의 모든 흐름을 통제하고 확장할 수 있습니다.
3. **Extensible:** `BaseLayer`, `BaseOptimizer`를 상속받아 나만의 최신 AI 논문 아이디어를 가장 빠르게 테스트하세요.

## 📦 설치 방법

```bash
pip install novagrad
```
또는
```bash
pip install git+https://github.com/yeunzu/NovaGrad.git
```

## ⚡ 퀵 스타트 (Quick Start)
가장 간단한 다층 퍼셉트론(MLP)으로 MNIST 데이터를 학습하는 예제입니다.
```python
import novagrad
from novagrad.models import MyModel
from novagrad.layers import Affine, Relu, Softmax
from novagrad.optimizers import Adam
from novagrad.losses import CrossEntropyError

# 1. 백엔드 자동 감지 및 초기화 (JAX or MLX)
be = novagrad.Backend()

# 2. 모델 조립
model = MyModel(backend=be)
model.add_layer(Affine(input_size=784, output_size=256, optimizer=Adam, backend=be))
model.add_layer(Relu(backend=be))
model.add_layer(Affine(input_size=256, output_size=10, optimizer=Adam, backend=be))
model.add_layer(Softmax(backend=be))

model.set_loss_func(CrossEntropyError(backend=be))

# 3. 학습 시작!
# trainer.fit() 등을 이용해 초고속 학습을 경험하세요.
```

## 📖 공식 문서 (Documentation)
레이어별 상세 파라미터, 데이터 증강(Albumentations) 사용법, 그리고 커스텀 모델 구축 가이드는 아래 공식 Notion 페이지에서 확인하실 수 있습니다.

👉 [NovaGrad 공식 사용 설명서 (Notion 링크)](https://app.notion.com/p/Home-NovaGrad-Documentation-3944bb67167a80bf845af6676ad69893?source=copy_link)

## 🤝 기여하기 (Contributing)
NovaGrad는 전 지구적 문제를 해결하기 위한 오픈소스 프로젝트입니다. 버그 리포트, 기능 제안, Pull Request를 언제나 환영합니다!

