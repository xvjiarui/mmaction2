from torch import nn

from ..registry import NECKS


@NECKS.register_module()
class PseudoNeck(nn.Module):

    def __init__(self, out_index):
        super(PseudoNeck, self).__init__()
        self.out_index = out_index

    def forward(self, x):
        return x[self.out_index]

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        pass

    def extra_repr(self) -> str:
        return f'out_index={self.out_index}'
