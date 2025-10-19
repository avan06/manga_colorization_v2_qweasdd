# manga_color_v2/networks/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint

from .extractor import SEResNeXt_Origin, BottleneckX_Origin

'''https://github.com/orashi/AlacGAN/blob/master/models/standard.py'''


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
    def __init__(self, ngf=64, use_checkpoint: bool = False,):
        super(Generator, self).__init__()
        
        self.use_checkpoint = use_checkpoint

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

        # --- Splitting Tunnel 4 ---
        self.tunnel4_pre = nn.Sequential(
            nn.Conv2d(1024 + 128, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.tunnel4_blocks = nn.ModuleList(
            [ResNeXtBottleneck(512, 512, cardinality=32, dilate=1) for _ in range(20)]
        )
        self.tunnel4_post = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        ) # 64x64 output

        # --- Splitting Tunnel 3 ---
        depth = 2
        self.tunnel3_pre = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        # Build tunnel3 blocks and insert Dropout2d in the middle (instead of nn.Dropout)
        tunnel3_modules = []
        tunnel3_modules.extend([ResNeXtBottleneck(256, 256, cardinality=32, dilate=1) for _ in range(depth)])
        tunnel3_modules.extend([ResNeXtBottleneck(256, 256, cardinality=32, dilate=2) for _ in range(depth)])
        # Insert spatial dropout (Dropout2d) as regularizer for conv feature maps
        # This layer has no effect during inference (model.eval()).
        tunnel3_modules.append(nn.Dropout2d(p=0.1))
        tunnel3_modules.extend([ResNeXtBottleneck(256, 256, cardinality=32, dilate=4) for _ in range(depth)])
        tunnel3_modules.extend([
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=2),
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=1)
        ])
        self.tunnel3_blocks = nn.ModuleList(tunnel3_modules)
        self.tunnel3_post = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        ) # 128x128 output

        # --- Splitting Tunnel 2 ---
        self.tunnel2_pre = nn.Sequential(
            nn.Conv2d(128 + 256 + 64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        # Build tunnel2 blocks and insert Dropout2d in the middle (instead of nn.Dropout)
        tunnel2_modules = []
        tunnel2_modules.extend([ResNeXtBottleneck(128, 128, cardinality=32, dilate=1) for _ in range(depth)])
        tunnel2_modules.extend([ResNeXtBottleneck(128, 128, cardinality=32, dilate=2) for _ in range(depth)])
        # Insert spatial dropout (Dropout2d) as regularizer for conv feature maps
        tunnel2_modules.append(nn.Dropout2d(p=0.1))
        tunnel2_modules.extend([ResNeXtBottleneck(128, 128, cardinality=32, dilate=4) for _ in range(depth)])
        tunnel2_modules.extend([
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=2),
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=1)
        ])
        self.tunnel2_blocks = nn.ModuleList(tunnel2_modules)
        self.tunnel2_post = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        ) # 256x256 output

        # --- Splitting Tunnel 1 (for consistency) ---
        self.tunnel1_pre = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.tunnel1_blocks = nn.ModuleList([
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=1),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=4),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)
        ])
        self.tunnel1_post = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        ) # This tunnel is not used in the final version, but the structure is retained

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
        
    def forward(self, sketch, return_feats=False): # Add return_feats parameter

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
        
        # --- Tunnel 4 ---
        out = self.tunnel4_pre(torch.cat([x4, aux_out], 1))
        # Use checkpoint only when in training mode (self.training is True) and the switch is enabled.
        if self.training and self.use_checkpoint:
            for block in self.tunnel4_blocks:
                out = checkpoint(block, out, use_reentrant=False)
        else:
            for block in self.tunnel4_blocks:
                out = block(out)
        out_tunnel4_result = self.tunnel4_post(out) # Store the final output of tunnel4

        # --- Tunnel 3 ---
        x = self.tunnel3_pre(torch.cat([out_tunnel4_result, x3], 1))
        if self.training and self.use_checkpoint:
            for block in self.tunnel3_blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.tunnel3_blocks:
                x = block(x)
        x = self.tunnel3_post(x)

        # --- Tunnel 2 ---
        x = self.tunnel2_pre(torch.cat([x, x2, x1], 1))
        if self.training and self.use_checkpoint:
            for block in self.tunnel2_blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.tunnel2_blocks:
                x = block(x)
        x = self.tunnel2_post(x)

        # Final output
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        
        # deconv_for_decoder uses the output of tunnel4
        decoder_output = self.deconv_for_decoder(out_tunnel4_result)
        
        # Add this conditional return statement
        if return_feats:
            return x, decoder_output, x4
        else:
            return x, decoder_output


class Colorizer(nn.Module):
    def __init__(self, use_checkpoint: bool = False,):
        super(Colorizer, self).__init__()
        
        self.generator = Generator(use_checkpoint=use_checkpoint,)
        
    def forward(self, x, return_feats: bool=False):
        if return_feats:
            fake, guide, sketch_feat = self.generator(x, return_feats=True)
            return fake, guide, sketch_feat
        else:
            fake, guide = self.generator(x)
            return fake, guide


# =================================================================================
#  The following is the newly added Discriminator-related code.
#  Inspired by NetD from AlacGAN (https://github.com/orashi/AlacGAN)
# =================================================================================


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
    def __init__(self, ndf=64, input_nc=3, sketch_feature_nc=1024, use_checkpoint: bool = False,):
        """
        ndf: Number of base feature maps in the discriminator
        input_nc: Number of channels in the input image (3 for a color image)
        sketch_feature_nc: Number of channels in the sketch feature map from the generator's Encoder
        """
        super(Discriminator, self).__init__()
        # Add a property to control whether to use checkpoint
        self.use_checkpoint = use_checkpoint

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
        self.output = nn.Sequential(
            # 1. Adaptive average pooling layer to convert any HxW input to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            # 2. Use a 1x1 convolution layer to produce the final 1-channel score
            nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0, bias=False)
        )


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
        score = self.output(fused_output) # (batch, 1, 1, 1)
        
        # Return a scalar score
        return score.view(-1)
