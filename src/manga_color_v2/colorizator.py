# manga_color_v2/colorizator.py

import cv2
import torch
import types
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from .networks.models import Colorizer
from .denoising.denoiser import FFDNetDenoiser
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


    def denoise_only(self, image_np: np.ndarray, sigma: int = 25, denoise_max_size: int = -1) -> np.ndarray:
        """
        Denoise a full-resolution image (numpy) using FFDNet and return RGB float32 [0,1].
        This method will preserve original size (it will internally resize/restore if needed).

        Args:
            image_np (np.ndarray): Input numpy image.
            sigma (int, optional): Denoising strength. Defaults to 25.
            denoise_max_size (int, optional): Maximum edge length allowed for denoising. -1 means no scaling. Defaults to -1 (no scaling).
        """
        if not isinstance(image_np, np.ndarray):
            raise TypeError("Input must be a numpy array")

        # Use the denoiser's helper which already handles resizing & conversion to RGB [0,1]
        denoised = self.denoiser.get_denoised_image(image_np, sigma=sigma, max_size=denoise_max_size)

        # Ensure dtype float32 and in [0,1]
        denoised = denoised.astype(np.float32)
        if denoised.max() > 1.2:
            denoised = denoised / 255.0
        denoised = np.clip(denoised, 0.0, 1.0)
        return denoised

        
    def set_image(self, image, size = 576, apply_denoise: bool = True, denoise_sigma: int = 25, denoise_max_size: int = -1, transform = ToTensor(),):
        if (size % 32 != 0):
            raise RuntimeError("size is not divisible by 32")
        
        if apply_denoise:
            # denoise_only returns float32 in [0,1]
            denoised_float = self.denoise_only(image, sigma=denoise_sigma, denoise_max_size=denoise_max_size)
            # Convert to uint8 for storage and consistent processing
            image = (denoised_float * 255.0).round().astype(np.uint8)
            # image = self.denoiser.get_denoised_image(image, sigma = denoise_sigma)
        
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
