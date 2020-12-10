from torch import nn

from ..registry import NECKS


@NECKS.register_module()
class SharedNeck(nn.Module):

    def __init__(self, in_index, out_index, strides=(1, 2, 1, 1)):
        super(SharedNeck, self).__init__()
        self.in_index = in_index
        self.out_index = out_index
        self.strides = strides

    def forward(self, x, backbone):
        backbone.switch_strides(self.strides)
        out = x[self.in_index]
        for i in range(self.in_index + 1, self.out_index + 1):
            layer_name = f'layer{i + 1}'
            res_layer = getattr(backbone, layer_name)
            out = res_layer(out)
        backbone.switch_strides()

        return out

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        pass

    def extra_repr(self) -> str:
        msg_str = f'in_index={self.in_index},'
        msg_str += f'out_index={self.out_index},'
        msg_str += f'strides={self.strides}'
        return msg_str
