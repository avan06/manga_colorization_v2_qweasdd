import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint

from .extractor import SEResNeXt_Origin, BottleneckX_Origin

'''https://github.com/orashi/AlacGAN/blob/master/models/standard.py'''

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out
    
class SelayerSpectr(nn.Module):
    def __init__(self, inplanes):
        super(SelayerSpectr, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = SpectralNorm(nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1))
        self.conv2 = SpectralNorm(nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))
            
        self.selayer = Selayer(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)
        
        x = self.shortcut.forward(x)
        return x + bottleneck

class SpectrResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(SpectrResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = SpectralNorm(nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_conv = SpectralNorm(nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False))
        self.conv_expand = SpectralNorm(nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))
            
        self.selayer = SelayerSpectr(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)
        
        x = self.shortcut.forward(x)
        return x + bottleneck
    
class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(FeatureConv, self).__init__()

        no_bn = True
        
        seq = []
        seq.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        seq.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)
    
class Generator(nn.Module):
    def __init__(self, ngf=64):
        super(Generator, self).__init__()
        
        self.use_checkpoint = False

        self.encoder = SEResNeXt_Origin(BottleneckX_Origin, [3, 4, 6, 3], num_classes= 370, input_channels=1)
        
        self.to0 =  self._make_encoder_block_first(5, 32)
        self.to1 = self._make_encoder_block(32, 64)
        self.to2 = self._make_encoder_block(64, 92)
        self.to3 = self._make_encoder_block(92, 128)
        self.to4 = self._make_encoder_block(128, 256)
        
        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
            nn.Tanh(),
        )

        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(512, 512, cardinality=32, dilate=1) for _ in range(20)])

        
        self.tunnel4 = nn.Sequential(nn.Conv2d(1024 + 128, 512, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel4,
                                     nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 64

        depth = 2
        tunnel = [ResNeXtBottleneck(256, 256, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=2),
                   ResNeXtBottleneck(256, 256, cardinality=32, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 128

        tunnel = [ResNeXtBottleneck(128, 128, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=2),
                   ResNeXtBottleneck(128, 128, cardinality=32, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(128 + 256 + 64, 128, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=4)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
                   ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.exit = nn.Sequential(nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(32, 3, kernel_size= 1, stride = 1, padding = 0))
        
        
    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )    
        
    def forward(self, sketch):

        x0 = self.to0(sketch)
        aux_out = self.to1(x0)
        aux_out = self.to2(aux_out)
        aux_out = self.to3(aux_out)
        
        # To use the encoder inside a checkpoint, we need to treat it as a whole
        # or ensure that x1, x2, x3, x4 do not need to retain gradients 
        # during the backward pass in the tunnel.
        # Here, the encoder output is directly fed into the tunnel,
        # so the gradients must be preserved.
        # Therefore, we execute the encoder first.
        x1, x2, x3, x4 = self.encoder(sketch[:, 0:1])
        
        # Use checkpoint only when in training mode (self.training is True) and the switch is enabled.
        if self.training and self.use_checkpoint:
            # Wrap computationally intensive tunnel layers with checkpoint
            # checkpoint takes a function and its inputs
            # We use lambda to simplify the call
            # Explicitly pass the parameter use_reentrant=False to suppress the warning and use the recommended implementation.
            out = checkpoint(self.tunnel4, torch.cat([x4, aux_out], 1), use_reentrant=False)
            x = checkpoint(self.tunnel3, torch.cat([out, x3], 1), use_reentrant=False)
            x = checkpoint(self.tunnel2, torch.cat([x, x2, x1], 1), use_reentrant=False)
        else:
            # In evaluation mode or when checkpointing is disabled,
            # execute the standard forward pass
            out = self.tunnel4(torch.cat([x4, aux_out], 1))
            x = self.tunnel3(torch.cat([out, x3], 1))
            x = self.tunnel2(torch.cat([x, x2, x1], 1))

        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        
        # deconv_for_decoder is also relatively resource-intensive,
        # but for now, we only checkpoint the tunnels to test the effect
        decoder_output = self.deconv_for_decoder(out)

        return x, decoder_output


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        
        self.generator = Generator()
        
    def forward(self, x, extractor_grad = False):
        fake, guide = self.generator(x)
        return fake, guide


# =================================================================================
#  The following is the newly added Discriminator-related code.
#  Inspired by NetD from AlacGAN (https://github.com/orashi/AlacGAN)
# =================================================================================

# You can directly use PyTorch's built-in spectral_norm without manually implementing the SpectralNorm class.
# However, to maintain consistency with the style of AlacGAN/Tag2Pix and to be able to apply it to non-Conv/Linear layers,
# an implementation of SelayerSpectr is provided here.

class SelayerSpectr(nn.Module):
    def __init__(self, inplanes):
        super(SelayerSpectr, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # Wrap convolutional layers with spectral_norm
        self.conv1 = spectral_norm(nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1))
        self.conv2 = spectral_norm(nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out


class ResNeXtBottleneck_D(nn.Module):
    """
    A ResNeXt Bottleneck module designed specifically for the discriminator.
    It differs slightly from the version in the generator, mainly in the
    activation function and network depth.
    Added Spectral Normalization and SELayer.
    """
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=8, dilate=1):
        super(ResNeXtBottleneck_D, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        
        # Use PyTorch's official spectral_norm function to wrap the convolutional layers directly
        self.conv_reduce = spectral_norm(nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_conv = spectral_norm(nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                                 groups=cardinality, bias=False))
        self.conv_expand = spectral_norm(nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.shortcut = nn.Sequential()
        if stride != 1:
            # Using AvgPool2d for downsampling, which is simpler and more effective than convolution
            self.shortcut.add_module('shortcut_pool', nn.AvgPool2d(2, stride=2))
        
        # Added SELayer
        self.selayer = SelayerSpectr(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        
        # Apply SELayer before the residual connection
        bottleneck = self.selayer(bottleneck)
        
        # Residual connection
        return self.shortcut(x) + bottleneck


class Discriminator(nn.Module):
    def __init__(self, ndf=64, input_nc=3, sketch_feature_nc=1024):
        """
        ndf: Number of base feature maps in the discriminator
        input_nc: Number of channels in the input image (3 for a color image)
        sketch_feature_nc: Number of channels in the sketch feature map from the generator's Encoder
        """
        super(Discriminator, self).__init__()
        # Add a property to control whether to use checkpoint
        self.use_checkpoint = False

        # Part 1: Process the color image with progressive downsampling
        self.feed = nn.Sequential(
            # input: (batch, 3, 512, 512)
            nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1, padding=3, bias=False),  # -> (batch, 64, 512, 512)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # -> (batch, 64, 256, 256)
            nn.LeakyReLU(0.2, True),

            ResNeXtBottleneck_D(ndf, ndf, cardinality=8, dilate=1),
            ResNeXtBottleneck_D(ndf, ndf, cardinality=8, dilate=1, stride=2),  # -> (batch, 64, 128, 128)
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False), # -> (batch, 128, 128, 128)
            nn.LeakyReLU(0.2, True),

            ResNeXtBottleneck_D(ndf * 2, ndf * 2, cardinality=8, dilate=1),
            ResNeXtBottleneck_D(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),  # -> (batch, 128, 64, 64)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False), # -> (batch, 256, 64, 64)
            nn.LeakyReLU(0.2, True),
            
            # Remove the last downsampling, change stride from 2 to 1
            # This makes the output feature map size 64x64, which matches sketch_feat (x4)
            ResNeXtBottleneck_D(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=1),
        )

        # Part 2: Fuse sketch features and image features
        # Input channels = image features (ndf*4) + sketch features (sketch_feature_nc)
        # AlacGAN's NetI outputs 512 channels, while our Encoder's x4 outputs 1024, so an adjustment is needed here.
        self.fuse = nn.Sequential(
            nn.Conv2d(ndf * 4 + sketch_feature_nc, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False), # -> (batch, 512, 64, 64)
            nn.LeakyReLU(0.2, True),
            ResNeXtBottleneck_D(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2), # -> (batch, 512, 32, 32)
            ResNeXtBottleneck_D(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2), # -> (batch, 512, 16, 16)
            ResNeXtBottleneck_D(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2), # -> (batch, 512, 8, 8)
        )
        
        # Final output layer
        # Since the output size of fuse becomes 8x8, the kernel size here needs to be adjusted to 8
        # to compress the 8x8 feature map into a 1x1 score
        self.output = nn.Conv2d(ndf * 8, 1, kernel_size=8, stride=1, padding=0, bias=False) # -> (batch, 1, 1, 1)

    def forward(self, color_image, sketch_features):
        """
        color_image: (batch, 3, H, W) The real or generated color image
        sketch_features: (batch, 1024, H/16, W/16) Sketch features (x4) extracted from the generator's encoder
        """
        # Process the image
        if self.training and self.use_checkpoint:
            # When checkpointing is enabled, use it to execute the computationally intensive parts
            image_feat = checkpoint(self.feed, color_image, use_reentrant=False)
            combined_feat = torch.cat([image_feat, sketch_features], 1)
            fused_output = checkpoint(self.fuse, combined_feat, use_reentrant=False)
        else:
            # Execute normally
            image_feat = self.feed(color_image)
        
            # Fuse features
            # Concatenate along the channel dimension using torch.cat
            # Now image_feat (64x64) and sketch_features (64x64) have matching dimensions
            combined_feat = torch.cat([image_feat, sketch_features], 1)
        
            # Process the fused features
            fused_output = self.fuse(combined_feat)
        
        # Get the final score
        score = self.output(fused_output)
        
        # Return a scalar score
        return score.view(-1)