# manga_color_v2/colorizator.py

import cv2
import torch
import types
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from .networks.models import Colorizer
from .denoising.denoiser import FFDNetDenoiser, normalize, variable_to_cv2_image
from .utils.utils import resize_pad

class MangaColorizator:
    def __init__(self, device, generator_path = 'networks/generator.zip', denoising_dir = 'networks/'):
        self.colorizer = Colorizer().to(device)
        state_dict = torch.load(generator_path, map_location = device, weights_only=False)
        self.colorizer.generator.load_state_dict(state_dict)
        self.colorizer = self.colorizer.eval()
        
        self.denoiser = FFDNetDenoiser(device, _weights_dir=denoising_dir)

        self.current_image = None
        self.current_hint = None
        self.current_pad = None

        self.device = device


    def denoise_only(self, image_np, sigma=25):
        """
        Denoises a full-resolution image and returns it at its original size.
        It handles resizing for the denoiser and resizing back.
        """
        if not isinstance(image_np, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        original_h, original_w, _ = image_np.shape

        # Perform denoising using the (now patched) method
        denoised_image = self.denoiser.get_denoised_image(image_np, sigma=sigma)

        # Resize the clean image back to its original dimensions if it was scaled down
        if denoised_image.shape[0] != original_h or denoised_image.shape[1] != original_w:
            if issubclass(denoised_image.dtype.type, np.floating):
                 denoised_image = (denoised_image * 255).astype(np.uint8)
            denoised_pil = Image.fromarray(denoised_image)
            restored_pil = denoised_pil.resize((original_w, original_h), Image.Resampling.LANCZOS)
            return np.array(restored_pil, dtype=np.float32) / 255.0
        else:
            return denoised_image

        
    def set_image(self, image, size = 576, apply_denoise = True, denoise_sigma = 25, transform = ToTensor()):
        if (size % 32 != 0):
            raise RuntimeError("size is not divisible by 32")
        
        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma = denoise_sigma)
        
        image, self.current_pad = resize_pad(image, size)
        self.current_image = transform(image).unsqueeze(0).to(self.device)
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2], self.current_image.shape[3]).float().to(self.device)
    
    def update_hint(self, hint, mask):
        '''
        Args:
           hint: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3], 3)
           mask: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3])
        '''
        
        if issubclass(hint.dtype.type, np.integer):
            hint = hint.astype('float32') / 255

        hint = (hint - 0.5) / 0.5
        hint = torch.FloatTensor(hint).permute(2, 0, 1)
        mask = torch.FloatTensor(np.expand_dims(mask, 0))

        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

    def colorize(self):
        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
            fake_color = fake_color.detach()

        result = fake_color[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5

        if self.current_pad[0] != 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] != 0:
            result = result[:, :-self.current_pad[1]]
            
        return result.numpy()
