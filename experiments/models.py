import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


####################################################################
######################       Resnet          #######################
####################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0, in_planes=64, stable_resnet=False):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        if stable_resnet:
            # Total number of blocks for Stable ResNet
            # https://arxiv.org/pdf/2002.08797.pdf
            L = 0
            for x in num_blocks:
                L+=x
            self.L = L
        else:
            self.L = 1
        
        self.masks = None

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes*8*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.L))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out) / self.temp
        
        return out
            

def resnet18(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    return model

def resnet34(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    return model

def resnet50(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    return model

def resnet101(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    return model

def resnet110(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    return model

def resnet152(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    return model


####################################################################
#######################   VGG    ###################################
####################################################################

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='CIFAR10', depth=19, cfg=None, affine=True, batchnorm=True,
                 init_weights=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.dataset = dataset
        if dataset == 'CIFAR10':
            num_classes = 10
        elif dataset == 'CIFAR100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    
def weights_init(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()


####################################################################
##################### MobileNet v2 #################################
####################################################################
def bottleneck_config(input_size):
    """ bottleneck configurations for 64x64 and 32x32 input"""
    if input_size == 32:
        config = {
            'expansion': [1, 6, 6, 6, 6, ],
            'planes': [16, 24, 32, 64, 96],
            'reps': [1, 2, 3, 4, 3],
            'stride': [1, 2, 2, 2, 1]
        }
    elif input_size == 64:
        config = {
            'expansion': [1, 6, 6, 6, 6, 6],
            'planes': [16, 24, 32, 64, 96, 160],
            'reps': [1, 2, 3, 4, 3, 3],
            'stride': [1, 2, 2, 2, 1, 2]
        }
    return config


class BottleneckResidualBlock(nn.Module):
    """ Bottleneck Residual Block
    1x1 Conv2D, 3x3 DWise, linear 1x1 conv2D
    """
    def __init__(self, expansion, in_planes, planes, stride=1):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, expansion*in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion*in_planes)
        self.dwise = nn.Conv2d(expansion*in_planes, expansion*in_planes, kernel_size=3, padding=1, stride=stride,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * in_planes)
        self.conv2 = nn.Conv2d(expansion*in_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.dwise(out)))
        out = self.bn3(self.conv2(out))
        return out


class ConvBNRelu6(nn.Module):
    """ Conv2 + BatchNorm + RelU6 activation """
    def __init__(self, in_planes, planes, kernel_size = 1, padding= 0):
        super(ConvBNRelu6, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))


class MobileNetV2(nn.Module):
    def __init__(self, dataset='CIFAR10', init_weights=True):
        super(MobileNetV2, self).__init__()
        self.dataset = dataset
        if dataset == 'CIFAR10':
            self.num_classes = 10
        elif dataset == 'CIFAR100':
            self.num_classes = 100
        elif dataset == 'tiny_imagenet':
            self.num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)

        self._make_layers()

        if init_weights:
            self.apply(weights_init)

    def forward(self, x):
        x_out = self.conv0(x)

        x_out = self.bottleneck1(x_out)

        x_out = self.bottleneck2(x_out)
        x_out = torch.add(x_out, self.bottleneck3(x_out))

        x_out = self.bottleneck4(x_out)
        x_out = torch.add(x_out, self.bottleneck5(x_out))
        x_out = torch.add(x_out, self.bottleneck6(x_out))

        x_out = self.bottleneck7(x_out)
        x_out = torch.add(x_out, self.bottleneck8(x_out))
        x_out = torch.add(x_out, self.bottleneck9(x_out))
        x_out = torch.add(x_out, self.bottleneck10(x_out))

        x_out = self.bottleneck11(x_out)
        x_out = torch.add(x_out, self.bottleneck12(x_out))
        x_out = torch.add(x_out, self.bottleneck13(x_out))

        if self.dataset == 'tiny_imagenet':
            x_out = self.bottleneck14(x_out)
            x_out = torch.add(x_out, self.bottleneck15(x_out))
            x_out = torch.add(x_out, self.bottleneck16(x_out))

        x_out = self.convEnd(x_out)
        x_out = self.avgpool(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.classifier(x_out)
        return x_out

    def _make_layers(self):
        self.conv0 = ConvBNRelu6(3, 32, kernel_size=3, padding=1)

        self.bottleneck1 = BottleneckResidualBlock(1, 32, 16, 1)  # 1 rep

        self.bottleneck2 = BottleneckResidualBlock(6, 16, 24, 2)  # 2 reps
        self.bottleneck3 = BottleneckResidualBlock(6, 24, 24, 1)

        self.bottleneck4 = BottleneckResidualBlock(6, 24, 32, 2)  # 3 reps
        self.bottleneck5 = BottleneckResidualBlock(6, 32, 32, 1)
        self.bottleneck6 = BottleneckResidualBlock(6, 32, 32, 1)

        self.bottleneck7 = BottleneckResidualBlock(6, 32, 64, 2)  # 4 reps
        self.bottleneck8 = BottleneckResidualBlock(6, 64, 64, 1)
        self.bottleneck9 = BottleneckResidualBlock(6, 64, 64, 1)
        self.bottleneck10 = BottleneckResidualBlock(6, 64, 64, 1)

        self.bottleneck11 = BottleneckResidualBlock(6, 64, 96, 1)  # 3 reps
        self.bottleneck12 = BottleneckResidualBlock(6, 96, 96, 1)
        self.bottleneck13 = BottleneckResidualBlock(6, 96, 96, 1)
        out_planes = 96

        if self.num_classes == 200:
            self.bottleneck14 = BottleneckResidualBlock(6, 96, 160, 2)  # 3 reps
            self.bottleneck15 = BottleneckResidualBlock(6, 160, 160, 1)
            self.bottleneck16 = BottleneckResidualBlock(6, 160, 160, 1)
            out_planes = 160

        self.convEnd = ConvBNRelu6(out_planes, 640)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Linear(640, self.num_classes, bias=True)

