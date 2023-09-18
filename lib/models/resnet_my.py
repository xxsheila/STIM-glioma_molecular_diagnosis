# import ptvsd
# ptvsd.enable_attach(address=('10.154.63.78',5690))
# ptvsd.wait_for_attach()

import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3_1d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5_1d(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv7_1d(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class BasicBlock3_1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3_1d, self).__init__()
        self.conv1 = conv3_1d(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3_1d(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5_1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5_1d, self).__init__()
        self.conv1 = conv5_1d(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5_1d(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # d = residual.shape[2] - out.shape[2]
        # out1 = residual[:,:,0:-d] + out
        out += residual
        out1 = self.relu(out)
        # out += residual

        return out1



class BasicBlock7_1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7_1d, self).__init__()
        self.conv1 = conv7_1d(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7_1d(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # d = residual.shape[2] - out.shape[2]
        # out1 = residual[:, :, 0:-d] + out
        out += residual
        out1 = self.relu(out)
        # out += residual

        return out1


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Multi_cov1d(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1]):
        super(Multi_cov1d, self).__init__()

        self.inplanes3 = [64, 64*3]
        self.inplanes5 = [64, 64*3]
        self.inplanes7 = [64, 64*3]

        # conv1x7, downsample 2
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # multi_layers_1, downsample 2
        self.layer1x3_1 = self._make_layer3(BasicBlock3_1d, 64, layers[0], stride=2, layer=1)
        self.layer1x5_1 = self._make_layer5(BasicBlock5_1d, 64, layers[0], stride=2, layer=1)
        self.layer1x7_1 = self._make_layer7(BasicBlock7_1d, 64, layers[0], stride=2, layer=1)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # multi_layers_2, downsample 2
        self.layer1x3_2 = self._make_layer3(BasicBlock3_1d, 64, layers[1], stride=2, layer=2)
        self.layer1x5_2 = self._make_layer5(BasicBlock5_1d, 64, layers[1], stride=2, layer=2)
        self.layer1x7_2 = self._make_layer7(BasicBlock7_1d, 64, layers[1], stride=2, layer=2)
        # self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def _make_layer3(self, block, planes, blocks, stride=1, layer=1):
        downsample = None
        inplanes3 = self.inplanes3[layer-1]
        if stride != 1 or inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes3, planes, stride, downsample))
        inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=1, layer=1):
        downsample = None
        inplanes5 = self.inplanes5[layer-1]
        if stride != 1 or inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes5, planes, stride, downsample))
        inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=1, layer=1):
        downsample = None
        inplanes7 = self.inplanes7[layer-1]
        if stride != 1 or inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes7, planes, stride, downsample))
        inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)

        out2_1 = self.layer1x3_1(out1)
        out2_2 = self.layer1x5_1(out1)
        out2_3 = self.layer1x7_1(out1)
        out2 = torch.cat([out2_1, out2_2, out2_3], dim=1)

        out3_1 = self.layer1x3_2(out2)
        out3_2 = self.layer1x5_2(out2)
        out3_3 = self.layer1x7_2(out2)
        out3 = torch.cat([out3_1, out3_2, out3_3], dim=1)
        return out3



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, patch_number=8):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes_0 = 64*3
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer0 = Multi_cov1d(input_channel=1)
        self.layer1 = self._make_layer_1(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.patch_number = patch_number
        self.conv1d_my = nn.Conv1d(in_channels=self.patch_number, out_channels=1, kernel_size=3, padding=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_1(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes_0 != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_0, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_0, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        N, C, L, H = x.size()
        x = x.permute(0,3,1,2).contiguous() #NHCL
        x = x.reshape(N*H, C, L)
        x = self.layer0(x)

        _, C1, L1 = x.size()
        x = x.reshape(N, H, C1, L1)
        x = x.transpose(1,2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        b, _ = x.shape

        x = x.view(
            int(b/self.patch_number), self.patch_number, -1
        )
        x = self.conv1d_my(x)
        x = x.squeeze()
        # print('fc param', self.fc.parameters().shape)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def get_model(**kwargs):
    # kwargs = {"num_classes":3}
    return resnet18(pretrained=False,**kwargs)

if __name__ == '__main__':
    kwargs = {"num_classes":3, "patch_number":4}
    model = resnet18(pretrained=False,**kwargs)
    data = torch.rand((64, 1, 256, 32))
    y = model(data)
    print(y.shape)