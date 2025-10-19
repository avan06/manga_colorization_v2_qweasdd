# manga_color_v2/colorizator.py

import os
import re
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
    Improved intelligent loader:
      - Accepts path or state-dict
      - Extracts nested 'state_dict' / 'netG_state_dict' etc.
      - Normalizes 'module.' prefixes
      - Remaps old monolithic tunnel.* -> tunnelX_pre / tunnelX_blocks / tunnelX_post
      - Aligns checkpoint tunnel_blocks indices to model tunnel_blocks indices by SEQUENTIAL MAPPING
        (this handles dropout insertion automatically because dropout has no params and thus no keys)
      - Verifies tensor shapes before assigning
      - Loads using strict=False and reports stats
    """
    # --- load source into `raw_sd` ---
    if isinstance(source, str): # If a path is passed
        if not os.path.exists(source):
            print(f"\n[Loader] Warning: Path does not exist, skipping. Path: {source}")
            return
        print(f"\n[Loader] Attempting to load weights from: {os.path.basename(source)}")
        raw = torch.load(source, map_location=device)
    elif isinstance(source, dict): # If an already loaded dictionary is passed
        raw = source
    else:
        print(f"\n[Loader] Error: Source must be a path or a dictionary, but got {type(source)}.")
        return

    # Extract nested state_dict if present
    dict_to_load = raw
    if isinstance(raw, dict):
        # prefer model-type-specific keys if possible
        potential_keys = ['state_dict', 'model', 'weight', 'netG_state_dict', 'netD_state_dict', 'generator', 'discriminator']
        for k in potential_keys:
            if k in raw and isinstance(raw[k], dict):
                dict_to_load = raw[k]
                print(f"[Loader] -> Extracted '{k}' from checkpoint.")
                break

    # copy to mutable dict
    sd = dict(dict_to_load)

    # normalize keys helper: remove common wrappers like 'module.' or 'model.' at start
    def normalize_keys(sd_in):
        out = OrderedDict()
        for k, v in sd_in.items():
            nk = k
            # remove leading 'module.' once
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            # remove leading 'model.' once
            if nk.startswith('model.'):
                nk = nk[len('model.'):]
            out[nk] = v
        return out

    sd = normalize_keys(sd)

    # target model state_dict
    target_sd = model.state_dict()

    # helper: gather indices for prefix like 'tunnel3_blocks' but accounting for full prefix
    # returns a mapping of full_prefix -> sorted index-list
    def find_prefixes_with_indices(keys, simple_name):
        """
        Scan keys and find candidate full prefixes that end with simple_name, returning
        dict: full_prefix -> sorted list of integer indices found after that prefix.
        Example: key "generator.tunnel3_blocks.0.conv_reduce.weight"
                 simple_name "tunnel3_blocks"
                 full_prefix = "generator.tunnel3_blocks"
                 index = 0
        """
        res = {}
        pat = re.compile(r'(^|\.)((?:[\w\-]+\.)*' + re.escape(simple_name) + r')\.(\d+)\.')
        for k in keys:
            m = pat.search(k)
            if m:
                full_pref = m.group(2)  # e.g. 'generator.tunnel3_blocks' or 'tunnel3_blocks'
                idx = int(m.group(3))
                res.setdefault(full_pref, set()).add(idx)
        # convert sets to sorted lists
        for p in list(res.keys()):
            res[p] = sorted(list(res[p]))
        return res

    # ---------- REMAP OLD MONOLITHIC TUNNELS ----------
    # If checkpoint contains keys like 'tunnel4.2.0.conv_reduce.weight' but not 'tunnel4_pre'
    def remap_old_monolithic(sd_in):
        sd_new = dict(sd_in)  # shallow copy
        for i in [4, 3, 2, 1]:
            old_base = f'tunnel{i}'
            # detect presence of old style (e.g., 'tunnel4.2.0.conv_reduce.weight')
            has_old = any(k.startswith(old_base + '.') for k in sd_in.keys())
            has_new_pre = any(old_base + '_pre' in k for k in sd_in.keys())
            if has_old and not has_new_pre:
                # remap conv at index 0 -> tunnelX_pre.0.*
                for k in list(sd_in.keys()):
                    if k.startswith(f'{old_base}.0.'):
                        newk = k.replace(f'{old_base}.0.', f'{old_base}_pre.0.')
                        sd_new[newk] = sd_in[k]
                        # optionally remove old key later
                # remap inner blocks at index 2 -> tunnelX_blocks.<b>.
                for k in list(sd_in.keys()):
                    m = re.match(re.escape(old_base) + r'\.2\.(\d+)\.(.*)', k)
                    if m:
                        bidx = int(m.group(1))
                        rest = m.group(2)
                        newk = f'{old_base}_blocks.{bidx}.{rest}'
                        sd_new[newk] = sd_in[k]
                # remap tail conv at index 3 -> tunnelX_post.0.
                for k in list(sd_in.keys()):
                    if k.startswith(f'{old_base}.3.'):
                        newk = k.replace(f'{old_base}.3.', f'{old_base}_post.0.')
                        sd_new[newk] = sd_in[k]
                # remove old keys that we remapped (safe: do this on sd_new after adding)
                for k in list(sd_in.keys()):
                    if k.startswith(old_base + '.'):
                        sd_new.pop(k, None)
                # move to next tunnel
        return sd_new

    sd = remap_old_monolithic(sd)

    # Normalize keys again (in case remap inserted 'module.' weirdly) - keep deterministic ordering
    sd = normalize_keys(sd)

    # ---------- Precompute candidate tunnel prefixes in source and model ----------
    tunnels_simple = ['tunnel4_blocks', 'tunnel3_blocks', 'tunnel2_blocks', 'tunnel1_blocks']
    src_prefix_map = {}
    tgt_prefix_map = {}
    for t in tunnels_simple:
        src_map = find_prefixes_with_indices(sd.keys(), t)
        tgt_map = find_prefixes_with_indices(target_sd.keys(), t)
        # pick first found prefix (if any) - but keep full map for fallback
        src_prefix_map[t] = src_map  # dict: full_prefix -> [indices]
        tgt_prefix_map[t] = tgt_map

    # ---------- Begin building new_state (only assign when shapes match) ----------
    new_state = OrderedDict()
    loaded_count = 0
    skipped_shape = []
    assigned_src_keys = set()

    # 1) Direct mapping attempts (exact key, remove/add module/model prefixes)
    def try_direct_mappings():
        nonlocal loaded_count
        for k_src, v_src in list(sd.items()):
            if k_src in target_sd:
                if tuple(v_src.shape) == tuple(target_sd[k_src].shape):
                    new_state[k_src] = v_src.clone()
                    loaded_count += 1
                    assigned_src_keys.add(k_src)
                else:
                    skipped_shape.append((k_src, v_src.shape, target_sd[k_src].shape))
                    assigned_src_keys.add(k_src)
            else:
                # try removing 'module.' or adding
                if k_src.startswith('module.'):
                    candidate = k_src[len('module.'):]
                    if candidate in target_sd and tuple(sd[k_src].shape) == tuple(target_sd[candidate].shape):
                        new_state[candidate] = sd[k_src].clone()
                        loaded_count += 1
                        assigned_src_keys.add(k_src)
                        continue
                candidate = 'module.' + k_src
                if candidate in target_sd and tuple(sd[k_src].shape) == tuple(target_sd[candidate].shape):
                    new_state[candidate] = sd[k_src].clone()
                    loaded_count += 1
                    assigned_src_keys.add(k_src)
                    continue
        # end direct mapping

    try_direct_mappings()

    # 2) Tunnel block sequential alignment mapping (robust Dropout handling)
    for simple_t in tunnels_simple:
        src_map = src_prefix_map.get(simple_t, {})
        tgt_map = tgt_prefix_map.get(simple_t, {})

        # iterate all possible prefix pairs (common_name in src_map vs model)
        for src_pref, src_indices in src_map.items():
            # Try to find best matching target prefix in model (same simple_t)
            # If multiple options exist in model, pick the one with largest intersection heuristically.
            best_tgt_pref = None
            best_score = -1
            for tgt_pref, tgt_indices in tgt_map.items():
                # score by min(len(src_indices), len(tgt_indices)) as heuristic
                score = min(len(src_indices), len(tgt_indices))
                if score > best_score:
                    best_score = score
                    best_tgt_pref = tgt_pref
            if best_tgt_pref is None:
                continue

            src_idxs = src_indices
            tgt_idxs = tgt_map[best_tgt_pref]  # model indices that HAVE params (dropout indices absent)
            if not src_idxs or not tgt_idxs:
                continue

            # Align sequentially: zip src_idxs -> tgt_idxs
            paired = list(zip(src_idxs, tgt_idxs))
            # If lengths differ, we'll map as many as possible (front alignment).
            for s_idx, t_idx in paired:
                s_prefix = f'{src_pref}.{s_idx}.'
                t_prefix = f'{best_tgt_pref}.{t_idx}.'
                # copy every key in sd that starts with s_prefix -> with t_prefix
                for k, v in sd.items():
                    if k.startswith(s_prefix):
                        new_key = k.replace(s_prefix, t_prefix, 1)
                        if new_key in target_sd:
                            if tuple(v.shape) == tuple(target_sd[new_key].shape):
                                new_state[new_key] = v.clone()
                                loaded_count += 1
                                assigned_src_keys.add(k)
                            else:
                                skipped_shape.append((k, v.shape, target_sd[new_key].shape))
                                assigned_src_keys.add(k)

    # 3) Additional heuristic transforms: insert 'module' at arbitrary positions (to handle CheckpointWrapper.module insertion)
    # We'll attempt limited attempts: try inserting 'module' between any two segments in src key and check match.
    def try_inserting_module():
        nonlocal loaded_count
        for k_src, v_src in list(sd.items()):
            if k_src in assigned_src_keys:
                continue
            parts = k_src.split('.')
            for i in range(1, len(parts)):
                candidate = '.'.join(parts[:i] + ['module'] + parts[i:])
                if candidate in target_sd and tuple(v_src.shape) == tuple(target_sd[candidate].shape):
                    new_state[candidate] = v_src.clone()
                    loaded_count += 1
                    assigned_src_keys.add(k_src)
                    break
                # also try removing 'module' if present in candidate
                if candidate.startswith('module.'):
                    c2 = candidate[len('module.'):]
                    if c2 in target_sd and tuple(v_src.shape) == tuple(target_sd[c2].shape):
                        new_state[c2] = v_src.clone()
                        loaded_count += 1
                        assigned_src_keys.add(k_src)
                        break

    try_inserting_module()

    # 4) Lastly, try matching by suffix patterns (fallback)
    def try_suffix_match():
        nonlocal loaded_count
        # build map suffix->target keys for quick lookup (limit suffix length to keep safe)
        suffix_map = {}
        for k in target_sd.keys():
            parts = k.split('.')
            for s_len in (3,4,5):  # try matching last 3-5 segments
                if len(parts) >= s_len:
                    suf = '.'.join(parts[-s_len:])
                    suffix_map.setdefault(suf, []).append(k)
        for k_src, v_src in list(sd.items()):
            if k_src in assigned_src_keys:
                continue
            parts_src = k_src.split('.')
            matched = False
            for s_len in (3,4,5):
                if len(parts_src) >= s_len:
                    suf = '.'.join(parts_src[-s_len:])
                    if suf in suffix_map:
                        for candidate in suffix_map[suf]:
                            if tuple(v_src.shape) == tuple(target_sd[candidate].shape):
                                new_state[candidate] = v_src.clone()
                                loaded_count += 1
                                assigned_src_keys.add(k_src)
                                matched = True
                                break
                if matched:
                    break

    try_suffix_match()

    # Actually load

    # Summary + load
    # Directly load weights using new_state; strict=False ignores mismatched keys and returns differences.
    # This call simultaneously loads weights and retrieves missing/unexpected keys.
    missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=False)

    print(f"[Loader] For model '{type(model).__name__}':")
    print(f"  - Successfully prepared {loaded_count} parameter tensors for loading (shape-checked).")

    if skipped_shape:
        print(f"  - Skipped {len(skipped_shape)} keys due to shape mismatch (examples: {skipped_shape[:3]}).")
        
    # Use the returned values from a single load_state_dict call for reporting
    # Count successfully loaded parameters
    successfully_loaded_count = sum(1 for k in new_state if k in target_sd and k not in unexpected_keys)
    print(f"  - Successfully loaded {successfully_loaded_count} / {len(target_sd)} parameter tensors.")

    if missing_keys:
        print(f"  - Ignored {len(missing_keys)} keys from target model (e.g., {missing_keys[:3]}...).")

    if unexpected_keys:
        print(f"  - Ignored {len(unexpected_keys)} keys from source file (e.g., {unexpected_keys[:3]}...).")
        
    # Check whether any weights were actually loaded
    if loaded_count == 0 and len(sd) > 0:
        example_src = next(iter(sd.keys()))
        example_tgt = next(iter(target_sd.keys()))
        print(f"  - [Warning] No compatible weights were assigned. Example source key: '{example_src}'. Example target key: '{example_tgt}'")

    return


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

        # In inference, we load weights directly into the generator submodule,
        # which is what the training script saves as the inference model.
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