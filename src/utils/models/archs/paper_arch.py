import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Identity):
    def __init__(self, drop_prob=0.):
        super().__init__()

# WindowAttention block
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# SwinTransformerBlock
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=7, 
                 shift_size=0, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = min(window_size, min(input_resolution))  
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, 
            num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        x_windows, padding = self.window_partition(x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H, W, padding)
        x = x.reshape(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            B, H, W, C = x.shape

        patch_h = H // self.window_size
        patch_w = W // self.window_size

        x = x.view(B, patch_h, self.window_size, patch_w, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows, (pad_h, pad_w)

    def window_reverse(self, windows, H, W, padding):
        pad_h, pad_w = padding
        B = int(windows.shape[0] / ((H + pad_h) / self.window_size * (W + pad_w) / self.window_size))
        x = windows.view(B, (H + pad_h) // self.window_size, (W + pad_w) // self.window_size, 
                         self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H + pad_h, W + pad_w, -1)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        return x

class WindowTransformerBlock2D(nn.Module):
    def __init__(self, dim, resolution, num_heads=2, window_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads
        )

    def forward(self, x):
        B, C, H, W = x.shape
        H, W = x.shape[2], x.shape[3]
        x_ = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B, H*W, C)
        x_ = self.attn(self.norm(x_))
        x_ = x_.view(B, H, W, C).permute(0, 3, 1, 2)  # back to (B, C, H, W)
        return x_
    
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
        self.use_transformer = [True, False, False, False, False, True]  # idx starts at 1

        self.conv1 = GhostModule(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, stride=1, idx=1)
        self.layer2 = self._make_layer(block, 32, stride=2, idx=2)
        self.layer3 = self._make_layer(block, 64, stride=2, idx=3)
        self.layer4 = self._make_layer(block, 128, stride=2, idx=4)
        self.layer5 = self._make_layer(block, 256, stride=2, idx=5)

        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, stride, idx):
        layers = []

        if self.use_transformer[idx]:
            res = 608 // (2 ** (idx + 1))  # 608 input with initial conv1 having stride 2
            layers.append(WindowTransformerBlock2D(self.in_planes, (res, res)))

        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        output.append(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); output.append(out)
        out = self.layer2(out); output.append(out)
        out = self.layer3(out); output.append(out)
        out = self.layer4(out); output.append(out)
        out = self.layer5(out); output.append(out)

        # for o in output:
        #     print(o.shape)

        return output

# Example usage:
def resnet32():
    return ResNet(BasicBlock, [10, 10, 10, 10, 10]) 

model = resnet32().to('cuda')

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

        self.check_input_shape(x)

        features = self.encoder(x)

        for ind in range(len(features)):
            features[ind]=features[ind]

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

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
        decoder_channels: List[int] = (256,128, 64,32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 3,
        activation: Optional[Union[str, callable]] = None,
    ):
        super().__init__()
        self.fusion=fusion
        self.encoder = model

        self.decoder = UnetDecoder(
            encoder_channels=((in_channels,16,32, 64, 128,256)),
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

