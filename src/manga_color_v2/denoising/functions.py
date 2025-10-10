"""
Functions implementing custom NN layers

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
from torch.autograd import Function, Variable

def concatenate_input_noise_map(input, noise_sigma):
    """
    Vectorized, device- and dtype-safe implementation.
    Keeps same channel ordering: top-left, top-right, bottom-left, bottom-right
    Output shape: (N, sca2*C + C, H//sca, W//sca) as original
    
    Implements the first layer of FFDNet. This function returns a
    torch.autograd.Variable composed of the concatenation of the downsampled
    input image and the noise map. Each image of the batch of size CxHxW gets
    converted to an array of size 4*CxH/2xW/2. Each of the pixels of the
    non-overlapped 2x2 patches of the input image are placed in the new array
    along the first dimension.

    Args:
        input: batch containing CxHxW images
        noise_sigma: the value of the pixels of the CxH/2xW/2 noise map
    """
    # noise_sigma is a list of length batch_size
    N, C, H, W = input.size()
    sca = 2
    sca2 = sca * sca
    Cout = sca2 * C

    Hout = H // sca
    Wout = W // sca
    
    # Fill the downsampled image with zeros
    # allocate on same device / dtype as input (avoid using string .type())
    downsampledfeatures = torch.zeros((N, Cout, Hout, Wout), device=input.device, dtype=input.dtype)

    # Build the CxH/2xW/2 noise_map (same device & dtype)
    noise_map = noise_sigma.view(N, 1, 1, 1).to(device=input.device, dtype=input.dtype).repeat(1, C, Hout, Wout)

    # Vectorized slices preserving order [(0,0),(0,1),(1,0),(1,1)]
    parts = [
        input[:, :, 0::2, 0::2],
        input[:, :, 0::2, 1::2],
        input[:, :, 1::2, 0::2],
        input[:, :, 1::2, 1::2],
    ]
    # concat along channel dim, but we need interleaved channel groups: each part has shape (N,C,Hout,Wout)
    downsampledfeatures = torch.cat(parts, dim=1)  # -> (N, 4*C, Hout, Wout)

    # concatenate noise map (noise first to match original implementation)
    return torch.cat((noise_map, downsampledfeatures), dim=1)

class UpSampleFeaturesFunction(Function):
    """
    Vectorized upsampling (inverse of concatenate_input_noise_map).
    forward: input is (N, Cin, Hin, Win) where Cin must be divisible by 4.
             returns (N, Cout, Hin*2, Win*2) with Cout = Cin//4
    backward: uses grad_output directly, no .data or Variable.

    Extends PyTorch's modules by implementing a torch.autograd.Function.
    This class implements the forward and backward methods of the last layer
    of FFDNet. It basically performs the inverse of
    concatenate_input_noise_map(): it converts each of the images of a
    batch of size CxH/2xW/2 to images of size C/4xHxW
    """
    @staticmethod
    def forward(ctx, input):
        # input: (N, Cin, Hin, Win), Cin must be divisible by 4
        N, Cin, Hin, Win = input.size()
        sca = 2
        sca2 = sca*sca
        assert (Cin % sca2 == 0), 'Invalid input dimensions: number of channels should be divisible by 4'
        
        Cout = Cin // sca2
        Hout = Hin * sca
        Wout = Win * sca

        # allocate result on same device/dtype
        result = torch.zeros((N, Cout, Hout, Wout), device=input.device, dtype=input.dtype)

        # Vectorized reassembly: take channel groups and scatter them into spatial grid
        # groups: 0 -> top-left, 1 -> top-right, 2 -> bottom-left, 3 -> bottom-right
        groups = [input[:, idx:Cin:sca2, :, :] for idx in range(sca2)]
        result[:, :, 0::2, 0::2] = groups[0]
        result[:, :, 0::2, 1::2] = groups[1]
        result[:, :, 1::2, 0::2] = groups[2]
        result[:, :, 1::2, 1::2] = groups[3]

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (N, C_out, H_out, W_out)
        N, Cg_out, Hg_out, Wg_out = grad_output.size()
        sca = 2
        sca2 = sca*sca
        Cg_in = sca2 * Cg_out
        Hg_in = Hg_out // sca
        Wg_in = Wg_out // sca

        # allocate grad_input on same device/dtype
        grad_input = torch.zeros((N, Cg_in, Hg_in, Wg_in), device=grad_output.device, dtype=grad_output.dtype)

        # gather the contributions
        grad_input[:, 0:Cg_in:sca2, :, :] = grad_output[:, :, 0::2, 0::2]
        grad_input[:, 1:Cg_in:sca2, :, :] = grad_output[:, :, 0::2, 1::2]
        grad_input[:, 2:Cg_in:sca2, :, :] = grad_output[:, :, 1::2, 0::2]
        grad_input[:, 3:Cg_in:sca2, :, :] = grad_output[:, :, 1::2, 1::2]

        return grad_input
    
# Alias functions
upsamplefeatures = UpSampleFeaturesFunction.apply
