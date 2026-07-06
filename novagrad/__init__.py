"""
NovaGrad
> A custom hybrid Deep Learning Framework with JAX and MLX backend.

[Official Links]
- Documentation (Notion): https://app.notion.com/p/Home-NovaGrad-Documentation-3944bb67167a80bf845af6676ad69893?source=copy_link
- GitHub Repository: https://github.com/yeunzu/NovaGrad
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("novagrad")
except PackageNotFoundError:
    __version__ = "unknown"

## =============
from .backend import Backend


from .models import (
    MyModel
)

from .albumentations import (
    MyCompose,
    GaussianNoise, 
    HorizontalFlip, 
    RandomBrightness, 
    RandomContrast
)

from .dataloaders import (
    DataLoader
)


from .optimizers import (
    SGD,
    Momentum,
    Adagrad, 
    RMSProp,
    Adam
)
from .activations import (
    Sigmoid, 
    Tanh, 
    Relu, 
    LeakyRelu, 
    Gelu
)
from .initializers import (
    Xavior, 
    He
)
from .losses import (
    CrossEntropyError, 
    MeanSquaredError, 
    BinaryCrossEntropy
)

from .layers import (
    Affine, 
    Dropout, 
    BatchNormalization, 
    Conv, 
    Flatten, 
    Pooling, 
    ResidualBlock
)

__all__ = [
    'Backend', 
    'MyModel', 
    'MyCompose', 'GaussianNoise', 'HorizontalFlip', 'RandomBrightness', 'RandomContrast', 
    'DataLoader', 
    'SGD', 'Momentum', 'Adagrad', 'RMSProp', 'Adam', 
    'Sigmoid', 'Tanh', 'Relu', 'LeakyRelu', 'Gelu', 
    'CrossEntropyError', 'MeanSquaredError', 'BinaryCrossEntropy', 
    'Xavior', 'He', 
    'Affine', 'Dropout', 'BatchNormalization', 'Conv', 'Flatten', 'Pooling', 'ResidualBlock', 
]

## =========

def about():
    """Prints information about the NovaGrad framework."""
    info_text = f"""
    ====================================================
    NovaGrad v{__version__}
    ====================================================
    
    * Author: Yeongju
    * Docs (Notion): https://app.notion.com/p/Home-NovaGrad-Documentation-3944bb67167a80bf845af6676ad69893?source=copy_link
    * GitHub: https://github.com/yeunzu/NovaGrad
    * License: MIT
    
    * Current Backend System: Ready to use!
    ====================================================
    """
    print(info_text)