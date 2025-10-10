# manga_color_v2/colorizator.py

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import math
from typing import Tuple
from collections import OrderedDict

from .networks.models import Colorizer
from .denoising.denoiser import FFDNetDenoiser

def load_weights_intelligently(model, source, device='cpu'):
    """
    A smarter, side-effect-free, and versatile weight loading function.
    - `source` can be a file path (str) or a pre-loaded state dictionary.
    - Intelligently extracts the correct state_dict.
    - Tries multiple key name translation strategies to handle 'module.' prefixes and model structure changes.
    - Only loads weights when both the key name and shape match.
    """
    if isinstance(source, str): # If a path is passed
        if not os.path.exists(source):
            print(f"\n[Loader] Warning: Path does not exist, skipping. Path: {source}")
            return
        print(f"\n[Loader] Attempting to load weights from: {os.path.basename(source)}")
        source_dict = torch.load(source, map_location=device)
    elif isinstance(source, dict): # If an already loaded dictionary is passed
        source_dict = source
    else:
        print(f"\n[Loader] Error: Source must be a path or a dictionary, but got {type(source)}.")
        return

    # --- Intelligent state_dict extraction ---
    target_key = None
    if isinstance(model, (Colorizer, nn.Module)): # Assuming Generator also inherits from nn.Module
        target_key = 'netG_state_dict'

    # Create a copy to operate on, to avoid modifying the original state
    dict_to_load = source_dict.copy()

    # Try to extract the relevant state_dict from the checkpoint
    if isinstance(dict_to_load, dict):
        # Prioritize using the key determined by the model type
        if target_key and target_key in dict_to_load:
            dict_to_load = dict_to_load[target_key]
            print(f"[Loader] -> Extracted '{target_key}' based on model type.")
        else:
            # If the specific key is not found, try common keys in order
            potential_keys = ['state_dict', 'model', 'weight', 'netG_state_dict', 'generator']
            for key in potential_keys:
                if key in dict_to_load:
                    dict_to_load = dict_to_load.get(key, dict_to_load) # Safely get, fallback to original
                    print(f"[Loader] -> Extracted generic key '{key}'.")
                    break
    
    target_model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    loaded_count = 0
    
    # --- Try multiple key name translation strategies ---
    for k_source, v_source in dict_to_load.items():
        k_target_found = None
        
        # Strategy 1: Direct match
        if k_source in target_model_dict:
            k_target_found = k_source
        
        # Strategy 2: Remove 'module.' prefix from the source
        if not k_target_found:
            k_candidate = k_source.replace('module.', '')
            if k_candidate in target_model_dict:
                k_target_found = k_candidate

        # Strategy 3: Add 'module.' prefix to the target key (to handle DataParallel)
        if not k_target_found:
            k_candidate = 'module.' + k_source
            if k_candidate in target_model_dict:
                k_target_found = k_candidate
        
        # Strategy 4: [Most Important] Handle internal '.module.' introduced by CheckpointWrapper
        # e.g., source: '...layer1.0.conv1...', target: '...layer1.0.module.conv1...'
        if not k_target_found:
            parts = k_source.split('.')
            for i in range(1, len(parts)):
                # Try inserting 'module' at different positions
                k_candidate = '.'.join(parts[:i] + ['module'] + parts[i:])
                if k_candidate in target_model_dict:
                    k_target_found = k_candidate
                    break

        # --- Strategy 5: Handle transition from old structure (single Sequential) to new structure (pre/blocks/post) ---
        if not k_target_found:
            k_candidate = k_source
            was_transformed = False
            for i in [4, 3, 2, 1]:
                # Rule 1: Transform the blocks part (the main transformation)
                # e.g., 'tunnel4.2.' -> 'tunnel4_blocks.'
                if f'tunnel{i}.2.' in k_candidate:
                    k_candidate = k_candidate.replace(f'tunnel{i}.2.', f'tunnel{i}_blocks.')
                    was_transformed = True
                
                # Rule 2: Transform the first Conv layer of the pre part
                # e.g., 'tunnel4.0.' -> 'tunnel4_pre.0.'
                elif f'tunnel{i}.0.' in k_candidate:
                    k_candidate = k_candidate.replace(f'tunnel{i}.0.', f'tunnel{i}_pre.0.')
                    was_transformed = True
                
                # Rule 3: Transform the Conv layer of the post part
                # e.g., 'tunnel4.3.' -> 'tunnel4_post.0.'
                elif f'tunnel{i}.3.' in k_candidate:
                    k_candidate = k_candidate.replace(f'tunnel{i}.3.', f'tunnel{i}_post.0.')
                    was_transformed = True
                
                if was_transformed:
                    break # Once a transformation is successful, break the loop for i

            if was_transformed and k_candidate in target_model_dict:
                k_target_found = k_candidate
        
        # If a matching key name is found, and the shape also matches, prepare to load it
        if k_target_found and v_source.shape == target_model_dict[k_target_found].shape:
            new_state_dict[k_target_found] = v_source
            loaded_count += 1

    # Safely load using strict=False
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"[Loader] For model '{type(model).__name__}':")
    print(f"  - Successfully loaded {loaded_count} / {len(target_model_dict)} parameter tensors.")
    if missing_keys:
        # Show only the first few as an example
        print(f"  - Ignored {len(missing_keys)} keys from target model (e.g., {missing_keys[:3]}...).")
    if unexpected_keys:
        print(f"  - Ignored {len(unexpected_keys)} keys from source file (e.g., {unexpected_keys[:3]}...).")
    
    # If after all strategies, the number of loaded weights is still 0, give a clear warning.
    if loaded_count == 0 and len(dict_to_load) > 0:
        print(f"  - [Warning] No compatible weights were loaded. Model remains as is.")
        print(f"  - Example key from source file: '{next(iter(dict_to_load.keys()))}'")
        print(f"  - Example key from target model: '{next(iter(target_model_dict.keys()))}'")


class MangaColorizator:
    def __init__(
        self,
        device,
        generator_path = 'networks/generator.zip',
        denoising_dir = 'networks/',
        target_resolution: int = 512,
        num_buckets: int = 10,
        dim_step: int = 64,
        max_aspect_ratio: float = 2.5
    ):
        self.colorizer = Colorizer().to(device)

        load_weights_intelligently(self.colorizer.generator, source=generator_path, device=device)
        
        self.colorizer = self.colorizer.eval()
        
        self.denoiser = FFDNetDenoiser(device, _weights_dir=denoising_dir)

        self.current_image = None
        self.current_hint = None
        self.current_pad = None
        # Add a new property to store the original image
        self.original_image = None

        self.device = device
        
        # --- 1. Create bucket list ---
        self.target_resolution = target_resolution
        self.buckets = self._create_aspect_ratio_buckets(target_resolution, num_buckets, dim_step, max_aspect_ratio)

    def _create_aspect_ratio_buckets(self, target_resolution=512, num_buckets=10, dim_step=8, max_aspect_ratio=2.5) -> list[tuple[int, int]]:
        """
        Generates a series of buckets that maintain an approximately constant total number of pixels, based on the target resolution.

        New features:
        - max_aspect_ratio: Limits the maximum aspect ratio of the generated buckets to avoid excessive VRAM peaks.

        Args:
            target_resolution (int): The target resolution, e.g., 512.
            num_buckets (int): The base number of buckets you want to create (the final count may vary slightly).
            dim_step (int): The dimension step size, should be a multiple of the network's total downsampling factor.
            max_aspect_ratio (float): The maximum allowed aspect ratio. For example, 2.5 means the longest side cannot exceed 2.5 times the shortest side.

        Returns:
            list[tuple[int, int]]: A list of (width, height) tuples.
        """
        target_area = target_resolution * target_resolution
        buckets = set()

        # Always include a standard square bucket
        buckets.add((target_resolution, target_resolution))
    
        # To record the number of skipped buckets
        skipped_count = 0

        # Start from a smaller aspect ratio and gradually approach the extreme values
        # We iterate over a range from 1 to max_aspect_ratio
        for i in range(num_buckets):
            # Use square root interpolation to make the distribution of aspect ratios more uniform on a logarithmic scale
            # This helps generate more buckets close to 1:1
            aspect_ratio_sqrt = 1.0 + (i / max(1, num_buckets - 1)) * (math.sqrt(max_aspect_ratio) - 1.0)
            aspect_ratio = aspect_ratio_sqrt ** 2
        
            # --- Handle horizontal buckets (width > height) ---
            w = math.sqrt(target_area * aspect_ratio)
            h = target_area / w
        
            # Align width and height to be multiples of dim_step
            w_rounded = round(w / dim_step) * dim_step
            h_rounded = round(h / dim_step) * dim_step
        
            if w_rounded > 0 and h_rounded > 0:
                current_ar = w_rounded / h_rounded
                # Check if the aspect ratio is within the allowed range
                if current_ar <= max_aspect_ratio:
                    buckets.add((w_rounded, h_rounded))
                else:
                    skipped_count += 1

            # --- Handle vertical buckets (height > width) ---
            # We can simply swap the width and height of the horizontal bucket to get the vertical one
            if h_rounded > 0 and w_rounded > 0:
                current_ar = h_rounded / w_rounded
                if current_ar <= max_aspect_ratio:
                    buckets.add((h_rounded, w_rounded))
                else:
                    # This check is theoretically redundant since the aspect ratio is the same, but kept for robustness
                    skipped_count += 1
            
        # Sort by area from smallest to largest and return
        sorted_buckets = sorted(list(buckets), key=lambda b: b[0] * b[1])

        return sorted_buckets

    def _find_best_bucket(self, width, height) -> Tuple[int, int]:
        """Find the best matching bucket size from self.buckets based on the aspect ratio of the input image."""
        aspect_ratio = width / height
        closest_bucket = min(
            self.buckets,
            key=lambda b: abs(b[0] / b[1] - aspect_ratio)
        )
        return closest_bucket


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
        

    def set_image(self, image, apply_denoise: bool = True, denoise_sigma: int = 25, denoise_max_size: int = -1, transform = ToTensor(),):
        # The 'size' parameter is removed, bucketing logic is used instead.
        
        # --- Denoising (optional) ---
        if apply_denoise:
            # denoise_only returns float32 in [0,1]
            denoised_float = self.denoise_only(image, sigma=denoise_sigma, denoise_max_size=denoise_max_size)
            # Convert to uint8 for storage and consistent processing
            image = (denoised_float * 255.0).round().astype(np.uint8)

        # Store a copy of the original image *before* any processing
        # Ensure it is in RGB format if it has 3 channels.
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.original_image = image.copy()
        else: # Handle grayscale or RGBA
            # For simplicity, convert to RGB for consistent L channel extraction later
            if len(image.shape) == 2: # Grayscale
                self.original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4: # RGBA
                self.original_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                 self.original_image = image.copy()
        
        # --- Resizing and Padding Logic ---
        h_orig, w_orig = image.shape[:2]

        # Find the best bucket for the image's aspect ratio
        bucket_w, bucket_h = self._find_best_bucket(w_orig, h_orig)

        # Calculate new dimensions to fit into the bucket while preserving aspect ratio
        scale = min(bucket_w / w_orig, bucket_h / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Ensure image is 3-channel for consistent processing
        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        
        if resized_image.shape[2] == 4:
            resized_image = resized_image[:, :, :3]

        # Calculate padding and store it
        pad_h = bucket_h - new_h
        pad_w = bucket_w - new_w
        self.current_pad = (pad_h, pad_w)
        
        # Pad the image (bottom and right) to the exact bucket dimensions
        # Use white padding, which is common for manga/sketches
        padded_image = np.pad(resized_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)

        # Convert to grayscale for the model's sketch input, and keep only one channel
        input_image = cv2.cvtColor(padded_image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        
        self.current_image = transform(input_image).unsqueeze(0).to(self.device)
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

    def colorize(self, use_original_l: bool = True):
        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
            fake_color = fake_color.detach()

        # Get model output and convert to numpy [0, 1] RGB
        result = fake_color[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5
        
        # Remove padding to get the result at the scaled resolution
        if self.current_pad[0] > 0:
            result = result[:-self.current_pad[0], :]
        if self.current_pad[1] > 0:
            result = result[:, :-self.current_pad[1]]
            
        result = result.numpy()

        if use_original_l:
            if self.original_image is None:
                raise RuntimeError("The original image is not set. Please call 'set_image' before 'colorize'.")

            # 1. Convert the model's colorized output (RGB float) to LAB
            # OpenCV works with uint8, so we scale it to [0, 255]
            model_output_rgb_uint8 = (result * 255).astype(np.uint8)
            model_output_lab = cv2.cvtColor(model_output_rgb_uint8, cv2.COLOR_RGB2LAB)
            
            # 2. Get the original L channel from the stored original image
            # Ensure the original image is uint8 for conversion
            original_image_uint8 = self.original_image
            if self.original_image.dtype != np.uint8:
                 original_image_uint8 = (self.original_image * 255).astype(np.uint8)

            original_lab = cv2.cvtColor(original_image_uint8, cv2.COLOR_RGB2LAB)
            original_l = original_lab[:, :, 0]

            # 3. Resize the 'a' and 'b' channels from the model output to match the original image's size
            target_h, target_w = original_l.shape
            resized_a = cv2.resize(model_output_lab[:, :, 1], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            resized_b = cv2.resize(model_output_lab[:, :, 2], (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            # 4. Merge the original L channel with the resized a and b channels
            final_lab = np.stack([original_l, resized_a, resized_b], axis=-1)

            # 5. Convert the final LAB image back to RGB and scale to [0, 1] float
            final_rgb_uint8 = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
            
            return final_rgb_uint8.astype(np.float32) / 255.0
        else:
            # If not using original L, return the model's direct output
            return result