#############################
#   @author: Nitin Rathi    #
#############################
import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG5': [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 256, 'A', 512, 512, 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class _quanFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tFactor):
        ctx.save_for_backward(input)
        max_w = input.abs().max()
        ctx.th = tFactor*max_w  # threshold
        output = input.clone().zero_()
        ctx.W = input[input.ge(ctx.th)+input.le(-ctx.th)].abs().mean()
        output[input.ge(ctx.th)] = ctx.W
        output[input.lt(-ctx.th)] = -ctx.W

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = ctx.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input, None


class quanConv2d(nn.Conv2d):
    def forward(self, input):
        tfactor_list = [0.05]
        quan_func = _quanFunc.apply
        weight = quan_func(self.weight, tfactor_list[0])
        output = F.conv2d(input, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output


class quanLinear(nn.Linear):

    def forward(self, input):
        tfactor_list = [0.05]
        quan_func = _quanFunc.apply
        weight = quan_func(self.weight, tfactor_list[0])
        output = F.linear(input, weight, self.bias)

        return output


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset='CIFAR10', kernel_size=3, dropout=0.2):
        super(VGG, self).__init__()

        self.dataset = dataset
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.features = self._make_layers(cfg[vgg_name])
        if vgg_name =='VGG11' and dataset=='CIFAR100':
          self.classifier = nn.Sequential(
            quanLinear(8192,1024,bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            quanLinear(1024,1024,bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            quanLinear(1024,labels,bias=False)
          )
        elif vgg_name == 'VGG5' and dataset != 'MNIST':
            self.classifier = nn.Sequential(
                quanLinear(512*4*4, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, labels, bias=False)
            )
        elif vgg_name != 'VGG5' and dataset != 'MNIST':
            self.classifier = nn.Sequential(
                quanLinear(512*2*2, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, labels, bias=False)
            )
        if vgg_name == 'VGG5' and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                quanLinear(128*7*7, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, labels, bias=False)
            )
        elif vgg_name != 'VGG5' and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                quanLinear(512*1*1, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, 4096, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                quanLinear(4096, labels, bias=False)
            )

        self._initialize_weights2()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights2(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
              #  n = m.weight.size(1)
              #  m.weight.data.normal_(0, 0.01)
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layers(self, cfg):
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3

        for x in cfg:
            stride = 1

            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [quanConv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
                           nn.ReLU(inplace=True)
                           ]
                layers += [nn.Dropout(self.dropout)]
                in_channels = x

        return nn.Sequential(*layers)


def test():
    for a in cfg.keys():
        if a == 'VGG5':
            continue
        net = VGG(a)
        x = torch.randn(2, 3, 32, 32)
        y = net(x)
        print(y.size())


    # For VGG5 change the linear layer in self. classifier from '512*2*2' to '512*4*4'
    # net = VGG('VGG5')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())
if __name__ == '__main__':
    test()
