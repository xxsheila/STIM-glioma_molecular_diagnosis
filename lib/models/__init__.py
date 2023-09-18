

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




# 模型入口,修改模型文件后,需要在这里添加相应的路径
from .resnet_my import get_model
from .thyroid import get_model
from .mobilenet import get_model
from .resnet_fusion import get_model
from .resnet_attention import get_model
from .resnet_attention_lstm import get_model
from .resnet_attention_lstm_compute import get_model