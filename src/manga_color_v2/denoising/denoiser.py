"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .models import FFDNet
from .utils import normalize, remove_dataparallel_wrapper

class FFDNetDenoiser:
    def __init__(self, _device, _sigma=25, _weights_dir='denoising/models/', _in_ch=3):
        """
        _device: torch.device or str (e.g. "cuda" or "cpu")
        _sigma: default noise level (0-255 scale)
        _weights_dir: directory where pretrained weights are stored
        _in_ch: number of input channels (3 for RGB, 1 for grayscale)
        """
        # unify device representation
        if isinstance(_device, str):
            self.device = torch.device(_device)
        elif isinstance(_device, torch.device):
            self.device = _device
        else:
            # accept None or other and fallback
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sigma = float(_sigma) / 255.0
        self.weights_dir = _weights_dir
        self.channels = int(_in_ch)

        # build model and load weights
        self.model = FFDNet(num_input_channels=self.channels)
        self.load_weights()
        self.model.to(self.device)
        self.model.eval()


    def load_weights(self):
        """
        Load model weights. Accepts CPU/GPU and removes any DataParallel wrapper if present.
        """
        weights_name = 'net_rgb.pth' if self.channels == 3 else 'net_gray.pth'
        weights_path = os.path.join(self.weights_dir, weights_name)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"FFDNet weights not found: {weights_path}")

        # Load state dict to CPU first (safer), then load into model and move model to device
        state_dict = torch.load(weights_path, map_location='cpu')
        # Remove DataParallel wrapper keys if present
        state_dict = remove_dataparallel_wrapper(state_dict)
        # Load into model (model will be moved to desired device later)
        self.model.load_state_dict(state_dict)

    def get_denoised_image(self, imorig, sigma=None, max_size=1200):
        """
        Denoise an input image and return RGB float32 image in [0,1].

        Args:
            imorig (np.ndarray): Input image as numpy array HxWxC (BGR or RGB).
            sigma (int, optional): Noise level in 0-255 scale. Defaults to self.sigma * 255.
            max_size (int, optional): The maximum size for the longest edge of the image
                                      before denoising. If the image is larger, it will
                                      be downscaled. Set to -1 to disable resizing.
                                      Defaults to 1200.
        """
        if sigma is not None:
            cur_sigma = float(sigma) / 255.0
        else:
            cur_sigma = float(self.sigma)

        # ensure 3-channel RGB
        if len(imorig.shape) < 3 or imorig.shape[2] == 1:
            imorig = np.repeat(np.expand_dims(imorig, 2), 3, 2)

        imorig = imorig[..., :3]  # keep 3 channels

        # --- Flexible resizing logic ---
        # Store the original dimensions for later restoration
        original_h, original_w = imorig.shape[:2]

        # Only scale down if max_size > 0 and the image dimensions exceed it
        # downscale extremely large images for denoiser memory reasons
        # The original code could result in a target size of (0, y) or (x, 0),
        # which causes an assertion error in cv2.resize.
        if max_size > 0 and max(original_h, original_w) > max_size:
            ratio = max(original_h, original_w) / float(max_size)
            new_w = max(2, int(original_w / ratio))
            new_h = max(2, int(original_h / ratio))
            
            # Ensure even dimensions, as FFDNet handles them better
            if new_w % 2 != 0: new_w += 1
            if new_h % 2 != 0: new_h += 1

            # Use INTER_AREA interpolation, which is best for shrinking images
            im_to_denoise = cv2.resize(imorig, (new_w, new_h), interpolation=cv2.INTER_AREA)
            was_resized = True
        else:
            # If not resizing, use the original image directly
            im_to_denoise = imorig
            was_resized = False
        
        # --- Subsequent denoising process is performed on im_to_denoise ---
        # normalize to [0,1] if necessary
        if im_to_denoise.max() > 1.2:
            im_to_denoise = normalize(im_to_denoise)
            
        # prepare tensor: (1,C,H,W) float32
        im_np = im_to_denoise.transpose(2, 0, 1).astype(np.float32)
        im_np = np.expand_dims(im_np, 0)
        im_t = torch.from_numpy(im_np).to(dtype=torch.float32, device='cpu')    # CPU tensor float32
        
        # handle odd sizes by padding last row/col via replicate (safer & faster)
        _, _, H, W = im_t.shape
        pad_h = 1 if (H % 2 == 1) else 0
        pad_w = 1 if (W % 2 == 1) else 0
        if pad_h or pad_w:
            # pad = (left, right, top, bottom)
            im_t = F.pad(im_t, (0, pad_w, 0, pad_h), mode='replicate')

        # move to device
        imnoisy = im_t.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # Create a per-sample noise vector matching batch size N
            N = imnoisy.shape[0]
            nsigma = torch.full((N,), float(cur_sigma), dtype=torch.float32, device=self.device)

            # Estimate noise and subtract it to the input image
            im_noise_estim = self.model(imnoisy, nsigma)
            outim = torch.clamp(imnoisy - im_noise_estim, 0.0, 1.0)

        if pad_h:
            outim = outim[:, :, :-1, :]

        if pad_w:
            outim = outim[:, :, :, :-1]
            
        # convert tensor to RGB float [0,1]
        # outim is (1, C, H, W) float on device
        outim_cpu = outim.detach().cpu().squeeze(0)  # (C,H,W)
        outim_np = outim_cpu.permute(1, 2, 0).numpy()  # (H,W,C) in [0,1] float32
        # clip just in case
        outim_np = np.clip(outim_np, 0.0, 1.0).astype(np.float32)
        # return RGB float [0,1] directly (caller expects this)
        
        # --- If the image was downscaled, scale it back up ---
        if was_resized:
            # Use INTER_CUBIC or INTER_LINEAR interpolation, which is better for enlarging images
            outim_np = cv2.resize(outim_np, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

        return outim_np
