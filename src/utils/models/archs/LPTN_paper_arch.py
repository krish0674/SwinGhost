import torch.nn as nn
import torch.nn.functional as F
import torch

# class Lap_Pyramid_Conv(nn.Module):
#     def __init__(self, num_high=3, device = torch.device('cuda')):
#         super(Lap_Pyramid_Conv, self).__init__()

#         self.num_high = num_high
#         self.device = device
#         self.kernel = self.gauss_kernel()

#     def gauss_kernel(self, channels=3):
#         kernel = torch.tensor([[1., 4., 6., 4., 1],
#                                [4., 16., 24., 16., 4.],
#                                [6., 24., 36., 24., 6.],
#                                [4., 16., 24., 16., 4.],
#                                [1., 4., 6., 4., 1.]])
#         kernel /= 256.
#         kernel = kernel.repeat(channels, 1, 1, 1)
#         kernel = kernel.to(self.device)
#         return kernel

#     def downsample(self, x):
#         return x[:, :, ::2, ::2]

#     def upsample(self, x):
#         cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
#         cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
#         cc = cc.permute(0, 1, 3, 2)
#         cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
#         cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
#         x_up = cc.permute(0, 1, 3, 2)
#         return self.conv_gauss(x_up, 4 * self.kernel)

#     def conv_gauss(self, img, kernel):
#         img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
#         out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
#         return out

#     def pyramid_decom(self, img):
#         current = img
#         pyr = []
#         for _ in range(self.num_high):
#             filtered = self.conv_gauss(current, self.kernel)
#             down = self.downsample(filtered)
#             up = self.upsample(down)
#             if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
#                 up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
#             diff = current - up
#             pyr.append(diff)
#             current = down
#         pyr.append(current)
#         return pyr

#     def pyramid_recons(self, pyr):
#         image = pyr[-1]
#         for level in reversed(pyr[:-1]):
#             up = self.upsample(image)
#             if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
#                 up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
#             image = up + level
#         return image

# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()

#         self.block = nn.Sequential(
#             nn.Conv2d(in_features, in_features, 3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_features, in_features, 3, padding=1),
#         )

#     def forward(self, x):
#         return x + self.block(x)

# class Trans_low(nn.Module):
#     def __init__(self, num_residual_blocks):
#         super(Trans_low, self).__init__()

#         model = [nn.Conv2d(3, 16, 3, padding=1),
#             nn.InstanceNorm2d(16),
#             nn.LeakyReLU(),
#             nn.Conv2d(16, 64, 3, padding=1),
#             nn.LeakyReLU()]

#         for _ in range(num_residual_blocks):
#             model += [ResidualBlock(64)]

#         model += [nn.Conv2d(64, 16, 3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(16, 3, 3, padding=1)]

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         out = x + self.model(x)
#         out = torch.tanh(out)
#         return out

# class Trans_high(nn.Module):
#     def __init__(self, num_residual_blocks, num_high=3):
#         super(Trans_high, self).__init__()

#         self.num_high = num_high

#         model = [nn.Conv2d(9, 64, 3, padding=1),
#             nn.LeakyReLU()]

#         for _ in range(num_residual_blocks):
#             model += [ResidualBlock(64)]

#         model += [nn.Conv2d(64, 1, 3, padding=1)]

#         self.model = nn.Sequential(*model)

#         for i in range(self.num_high):
#             trans_mask_block = nn.Sequential(
#                 nn.Conv2d(1, 16, 1),
#                 nn.LeakyReLU(),
#                 nn.Conv2d(16, 1, 1))
#             setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

#     def forward(self, x, pyr_original, fake_low):

#         pyr_result = []
#         mask = self.model(x)

#         for i in range(self.num_high):
#             mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
#             self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
#             mask = self.trans_mask_block(mask)
#             result_highfreq = torch.mul(pyr_original[-2-i], mask)
#             setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

#         for i in reversed(range(self.num_high)):
#             result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
#             pyr_result.append(result_highfreq)

#         pyr_result.append(fake_low)

#         return pyr_result

# class LPTNPaper(nn.Module):
#     def __init__(self, nrb_low=5, nrb_high=3, num_high=3, device = torch.device('cuda')):
#         super(LPTNPaper, self).__init__()

#         self.device = device
#         self.lap_pyramid = Lap_Pyramid_Conv(num_high, self.device)
#         trans_low = Trans_low(nrb_low)
#         trans_high = Trans_high(nrb_high, num_high=num_high)
#         self.trans_low = trans_low.to(self.device)
#         self.trans_high = trans_high.to(self.device)

#     def forward(self, real_A_full):

#         pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full)
#         fake_B_low = self.trans_low(pyr_A[-1])
#         real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
#         fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
#         high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
#         pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)
#         fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)

#         return pyr_A_trans,fake_B_full

from typing import Optional, Union, List
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import segmentation_models_pytorch as smp 


import torch
import torch.nn as nn
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ghost_net']
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y

def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size=3, stride=2, use_se=1):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        self.cfgs = cfgs

        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = 1280
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# def ghost_net(**kwargs):
#     """
#     Constructs a MobileNetV3-Large model
#     """
#     cfgs = [
#         # k, t, c, SE, s 
#         [3,  16,  16, 0, 1],
#         [3,  48,  24, 0, 2],
#         [3,  72,  24, 0, 1],
#         [5,  72,  40, 1, 2],
#         [5, 120,  40, 1, 1],
#         [3, 240,  80, 0, 2],
#         [3, 200,  80, 0, 1],
#         [3, 184,  80, 0, 1],
#         [3, 184,  80, 0, 1],
#         [3, 480, 112, 1, 1],
#         [3, 672, 112, 1, 1],
#         [5, 672, 160, 1, 2],
#         [5, 960, 160, 0, 1],
#         [5, 960, 160, 1, 1],
#         [5, 960, 160, 0, 1],
#         [5, 960, 160, 1, 1]
#     ]
#     return GhostNet(cfgs, **kwargs)


# __all__ = ['ResNet', 'resnet32']
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):

        super(BasicBlock, self).__init__()
        self.conv1 = GhostModule(in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

            elif option == 'B':
                self.shortcut = nn.Sequential(
                     GhostModule(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
#         self.layer0 = self._make_layer(block, 3, num_blocks[0], stride=2)
        self.conv1 = GhostModule(1, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 256, num_blocks[4], stride=2)

        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output=[]
        output.append(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        output.append(out)
        out = self.layer2(out)
        output.append(out)
        out = self.layer3(out)
        output.append(out)

        out = self.layer4(out)
        output.append(out)

        out = self.layer5(out)
        output.append(out)

        return output
    
def resnet32():
    return ResNet(BasicBlock, [10, 10, 10, 10, 10])


model = resnet32() 

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)
                         
class SegmentationModel(torch.nn.Module):
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = 32
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x,y=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        # if self.fusion == True:
        #     features1 = self.encoder2(x)
            
        #     f1 = features[-1]
        #     #f2 = features1[-1]
            
        for ind in range(len(features)):
            features[ind]=features[ind]
        #     #     # features[ind] = (features[ind]+features1[ind])/2
        #     #     # features[ind] = features1[ind]
        #     #     features[ind] = torch.maximum(features[ind],features1[ind])
        #     #     # features[ind] = torch.cat((features[ind],features1[ind]),1)
    
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        # if self.contrastive_head1 is not None:
        #     f1= self.contrastive_head1(f1)
        #     f2= self.contrastive_head2(f2)
        #     return masks, f1,  f2
        return masks

    @torch.no_grad()
    def predict(self, x, y=None):
        if self.training:
            self.eval()
        if self.contrastive_head1 is not None:
            x, _, _ = self.forward(x,y)
            return x
        if y is not None:
            x = self.forward(x,y)
            return x
        x = self.forward(x)

        return x
                         
class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        fusion:bool=True,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 3,
        activation: Optional[Union[str, callable]] = None,
        contrastive: bool = False,
    ):
        super().__init__()
        self.fusion=fusion
        self.encoder = model
        #self.encoder2 = model1

        self.decoder = UnetDecoder(
            encoder_channels=((3,16,32, 64, 128, 256)),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        self.name = "u-{}".format(encoder_name)
        self.initialize()





