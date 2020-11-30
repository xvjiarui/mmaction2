from .base import BaseHead
from .i3d_head import I3DHead
from .moco_head import MoCoHead
from .sim_siam_head import DenseSimSiamHead, SimSiamHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .uvc_head import UVCHead
from .walker_head import WalkerHead
from .walker_head_v2 import WalkerHeadV2

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'WalkerHead', 'UVCHead', 'MoCoHead', 'WalkerHeadV2', 'SimSiamHead',
    'DenseSimSiamHead'
]
