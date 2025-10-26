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
        # store the original image
        self.original_image = None
        # Store the calculated scan artifact SCOR
        self.scan_artifact_score = 0.0

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

    def set_image(self, image, apply_denoise: bool = True, denoise_sigma: int = 25, denoise_max_size: int = -1,
                      descreen_for_model_input: str = 'auto', descreen_strength_for_input: float = 0.9,
                      pre_processing_descreen_threshold: float = 1.35,
                      transform=ToTensor()):
        """
        Set the current image for colorization, with optional denoising and descreening preprocessing.
        # The 'size' parameter is removed, bucketing logic is used instead.
        Args:
            image: numpy.ndarray with shape (H, W, C) or (H, W)
            apply_denoise (bool): Whether to apply denoising before processing.
            denoise_sigma (int): Denoising strength for FFDNet.
            denoise_max_size (int): Maximum edge length for denoising. -1 means no resizing.
            descreen_for_model_input (str): 'auto', 'force_on', or 'force_off' for descreening model input.
            descreen_strength_for_input (float): Strength of descreening for model input (0.0 to 1.0).
        """
        # --- Denoising (optional, FFDNet) ---
        if apply_denoise:
            # denoise_only returns float32 in [0,1]
            denoised_float = self.denoise_only(image, sigma=denoise_sigma, denoise_max_size=denoise_max_size)
            # Convert to uint8 for storage and consistent processing
            image = (denoised_float * 255.0).round().astype(np.uint8)
        
        # 1. Unconditionally save the most original, unprocessed image for final composition
        # Store a copy of the original image *before* any processing
        # Ensure it is in RGB format if it has 3 channels.
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.original_image = image.copy()
        else: # Handle grayscale or RGBA
            # For simplicity, convert to RGB for consistent L channel extraction later
            if len(image.shape) == 2: # Grayscale
                self.original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4: # RGBA
                self.original_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                self.original_image = image.copy()

        # 2. Create a copy of the image specifically for the model and preprocess it
        image_for_model = self.original_image.copy()
        
        apply_input_descreening = False
        if descreen_for_model_input == 'force_on':
            print("[Info] Performing descreening preprocessing for model input...")
            apply_input_descreening = True
        elif descreen_for_model_input == 'auto':
            # Extract the L channel from the copy for detection and store the result in self.scan_artifacts_detected
            l_for_detect = cv2.cvtColor(image_for_model, cv2.COLOR_RGB2GRAY)
            self.scan_artifact_score = self._calculate_scan_artifact_score(l_for_detect)
            if self.scan_artifact_score > pre_processing_descreen_threshold:
                print(f"[Info] Pre-processing descreen triggered (Score {self.scan_artifact_score:.2f} > Strict Threshold {pre_processing_descreen_threshold}).")
                apply_input_descreening = True
            else:
                print(f"[Info] Pre-processing descreen skipped (Score {self.scan_artifact_score:.2f} <= Strict Threshold {pre_processing_descreen_threshold}).")
        
        if apply_input_descreening:
            # --- Use the LAB method to separate and recombine the L channel ---
            # 1. Convert image_for_model to LAB color space
            lab_image_for_model = cv2.cvtColor(image_for_model, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_image_for_model)

            # 2. Perform texture reduction on the L channel
            cleaned_l = self.reduce_texture_camera_raw_style(l_channel, strength=descreen_strength_for_input, radius=4, iterations=3)

            # 3. Merge the cleaned L channel with the original a and b channels
            #    (For grayscale images, the a and b channels are typically neutral gray (128), and preserving them is standard practice)
            cleaned_lab = cv2.merge([cleaned_l, a_channel, b_channel])
            
            # 4. Convert the LAB image back to RGB for subsequent processing
            image_for_model = cv2.cvtColor(cleaned_lab, cv2.COLOR_LAB2RGB)
        
        # --- All subsequent processing is based on image_for_model ---
        # --- Resizing and Padding Logic ---
        h_orig, w_orig = image_for_model.shape[:2]

        # Find the best bucket for the image's aspect ratio
        bucket_w, bucket_h = self._find_best_bucket(w_orig, h_orig)

        # Calculate new dimensions to fit into the bucket while preserving aspect ratio
        scale = min(bucket_w / w_orig, bucket_h / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        
        # Resize the image
        resized_image = cv2.resize(image_for_model, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
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

        print(f"[MangaColorizator] Image prepared for model. Original size: ({w_orig}, {h_orig}), Resized to: ({new_w}, {new_h}), Padded to bucket: ({bucket_w}, {bucket_h}). Final tensor shape: {self.current_image.shape}")

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

    def _calculate_scan_artifact_score(self, l_channel_uint8: np.ndarray, center_mask_ratio: float = 0.15) -> float:
        """
        Detects periodic scan artifacts in the image using Fast Fourier Transform (FFT).
        Calculates a score representing the likelihood of periodic scan artifacts.
        Instead of returning a boolean, it returns the ratio of the peak to the average,
        which can then be compared against different thresholds.

        Args:
            l_channel_uint8 (np.ndarray): The original L channel (uint8) to analyze.
            center_mask_ratio (float): The radius ratio of the central area of the spectrum to be masked.

        Returns:
            float: The calculated artifact score. A higher score means more likely to have artifacts.
                   Returns 0.0 if calculation is not possible.
        """
        # 1. Perform Fourier Transform
        f = np.fft.fft2(l_channel_uint8)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)

        rows, cols = l_channel_uint8.shape
        crow, ccol = rows // 2, cols // 2

        # 2. Create a uint8 mask to ignore the normal signal in the center of the spectrum (low frequencies)
        mask_radius = int(min(crow, ccol) * center_mask_ratio)
        
        # Change dtype from bool to np.uint8
        mask = np.ones(magnitude_spectrum.shape, dtype=np.uint8)
        
        # Change the drawing color from False to 0
        cv2.circle(mask, (ccol, crow), mask_radius, 0, -1)

        # 3. Analyze the high-frequency spectrum
        # Search for anomalous peaks in the non-central region
        # Convert the uint8 mask to a boolean type for indexing
        high_freq_spectrum = magnitude_spectrum[mask.astype(bool)]

        if high_freq_spectrum.size == 0:
            return 0.0

        # Calculate the mean and maximum values
        mean_val = np.mean(high_freq_spectrum)
        max_val = np.max(high_freq_spectrum)

        # 4. Return the score instead of a boolean
        # If mean_val is zero, return 0.0 to avoid division by zero
        if mean_val > 0:
            score = max_val / mean_val
            # print(f"[Info] Calculated Scan Artifact Score: {score:.2f}")
            return score
        else:
            # print("[Info] No significant high-frequency components found.")
            return 0.0

    def reduce_texture_camera_raw_style(self, l_channel_uint8: np.ndarray,
                                        strength: float = 0.6,
                                        radius: int = 3,
                                        iterations: int = 2,
                                        edge_protect: bool = True,
                                        edge_sensitivity: float = 0.8) -> np.ndarray:
        """
        Approximate Camera Raw 'Texture' reduction on a single L channel.

        Args:
            l_channel_uint8: 2D numpy array (H, W), dtype=uint8, range 0..255.
            strength: how strongly to remove texture (0.0..1.0).
                      0.0 = keep original, 1.0 = remove fine detail completely.
            radius: filter radius hint; larger values affect larger-scale texture.
            iterations: number of iterative bilateral passes to better separate base/detail.
            edge_protect: if True, preserve edges (text/lines) by blending original nearby edges.
            edge_sensitivity: how strongly to preserve edges (0..1). Higher = more preservation.

        Returns:
            2D numpy array dtype=uint8, same shape as input.
        """
        if not isinstance(l_channel_uint8, np.ndarray):
            raise TypeError("l_channel_uint8 must be a numpy array")
        if l_channel_uint8.ndim != 2:
            raise ValueError("Input must be a 2D array (single L channel)")
        if l_channel_uint8.dtype != np.uint8:
            raise TypeError("Input must be dtype=uint8 (0..255)")

        # Convert to float32 for processing
        orig = l_channel_uint8.astype(np.float32)

        # Normalize to 0..255 (already), but keep as float
        base = orig.copy()

        # Iterative bilateral filtering (a rolling-guidance style)
        # Use sigmaColor scaled with radius and iterations to better remove mid-frequency texture.
        # d=0 lets OpenCV compute kernel from sigmas.
        sigma_space = max(1.0, radius)
        # sigmaColor controls how much colors (here brightness) mix — scale with radius and iterations
        sigma_color = max(10.0, radius * 8.0)

        for i in range(iterations):
            # progressively increase sigma_color to smooth more aggressively on later passes
            sc = sigma_color * (1.0 + i * 0.6)
            base = cv2.bilateralFilter(base.astype(np.float32), d=0,
                                       sigmaColor=sc, sigmaSpace=sigma_space)

        # Detail layer
        detail = orig - base

        # Reduce detail (this is the key "texture" reduction step)
        processed = base + detail * (1.0 - float(np.clip(strength, 0.0, 1.0)))

        # Edge protection mask to avoid blurring text/strokes
        if edge_protect:
            # Compute gradient magnitude via Laplacian (sensitive to thin strokes)
            lap = cv2.Laplacian(orig, ddepth=cv2.CV_32F, ksize=3)
            mag = np.abs(lap)

            # Normalize magnitude to 0..1
            # Use robust scaling: subtract median and scale by (mean absolute deviation)
            med = np.median(mag)
            mad = np.median(np.abs(mag - med)) + 1e-6
            norm = (mag - med) / (mad * 3.0)  # dividing to compress extremes
            norm = np.clip(norm, 0.0, 1.0)

            # Threshold soft mask and smooth it so edges have soft transitions
            # edge_sensitivity controls thresholding: higher preserves more
            thresh = 0.08 * (1.0 + (1.0 - edge_sensitivity) * 2.5)
            mask = (norm > thresh).astype(np.float32)

            # Dilate mask a bit to fully cover strokes/text (kernel size scales with radius)
            ksize = max(1, int(radius))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1 + 2*ksize, 1 + 2*ksize))
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Blur mask to make protection soft
            mask = cv2.GaussianBlur(mask, (1 + 2*ksize, 1 + 2*ksize), sigmaX=ksize + 0.1)

            # Final blend: where mask ~1 use original more, where mask ~0 use processed more.
            # edge_sensitivity also scales how strongly we favor the original at edges.
            alpha = np.clip(mask * float(np.clip(edge_sensitivity, 0.0, 1.0)), 0.0, 1.0)
            out = processed * (1.0 - alpha) + orig * alpha
        else:
            out = processed

        # Ensure valid range and convert back to uint8
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def _merge_with_model_l_pyramid(self, orig_l_uint8: np.ndarray, model_rgb_uint8: np.ndarray, levels:int=3, blend:float=0.9) -> np.ndarray:
        """
        Laplacian pyramid fusion:
        - Replace high-frequency layers of original L with those from model's L (upsampled).
        - levels: pyramid depth (1..4). higher => more aggressive replacement.
        - blend: how strongly to prefer model high-freq (0..1). 1.0 = full replace.
        Returns uint8 L in 0..255.
        """
        # ensure same size (we will upsample model to original size first)
        h, w = orig_l_uint8.shape[:2]
        # model -> L
        model_lab = cv2.cvtColor(model_rgb_uint8, cv2.COLOR_RGB2LAB)
        model_l = model_lab[:, :, 0]
        model_l_up = cv2.resize(model_l, (w, h), interpolation=cv2.INTER_CUBIC)

        # build Gaussian pyramids
        def gaussian_pyr(img, n):
            gp = [img.astype(np.float32)]
            for i in range(n):
                img = cv2.pyrDown(img)
                gp.append(img.astype(np.float32))
            return gp

        def laplacian_from_gaussian(gp):
            lp = []
            for i in range(len(gp)-1):
                g_up = cv2.pyrUp(gp[i+1])
                # if size mismatch, resize
                if g_up.shape != gp[i].shape:
                    g_up = cv2.resize(g_up, (gp[i].shape[1], gp[i].shape[0]), interpolation=cv2.INTER_LINEAR)
                lp.append(gp[i] - g_up)
            lp.append(gp[-1])
            return lp

        gp_orig = gaussian_pyr(orig_l_uint8, levels)
        gp_model = gaussian_pyr(model_l_up, levels)
        lp_orig = laplacian_from_gaussian(gp_orig)
        lp_model = laplacian_from_gaussian(gp_model)

        # fuse: for higher-frequency levels (0..levels-1) mix with model
        fused_lp = []
        for i in range(len(lp_orig)):
            if i < levels:  # high-frequency levels
                fused = lp_orig[i] * (1.0 - blend) + lp_model[i] * (blend)
            else:
                fused = lp_model[i] * (blend) + lp_orig[i] * (1.0 - blend)
            fused_lp.append(fused)

        # reconstruct
        img_recon = fused_lp[-1]
        for i in range(len(fused_lp)-2, -1, -1):
            img_recon = cv2.pyrUp(img_recon)
            if img_recon.shape != fused_lp[i].shape:
                img_recon = cv2.resize(img_recon, (fused_lp[i].shape[1], fused_lp[i].shape[0]), interpolation=cv2.INTER_LINEAR)
            img_recon = img_recon + fused_lp[i]

        img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)
        return img_recon

    def _clean_l_with_ai_guidance(self, original_l_uint8: np.ndarray, ai_l_uint8: np.ndarray, radius: int = 4, eps: float = 200.0) -> np.ndarray:
            """
            Cleans the original L channel using a guided filter, with the AI-generated L channel as the guide.

            Args:
                original_l_uint8 (np.ndarray): The original L channel with scan artifacts.
                ai_l_uint8 (np.ndarray): The AI-generated, smooth L channel to be used as the guide.
                radius (int): The radius for the guided filter.
                eps (float): The regularization parameter (smoothing strength) for the guided filter. A higher value results in stronger smoothing.

            Returns:
                np.ndarray: The cleaned L channel (uint8).
            """
            try:
                import cv2.ximgproc
            except ImportError:
                raise ImportError("Guided filter requires 'opencv-contrib-python'. Please run 'pip install opencv-contrib-python'")
        
            # The guided filter requires the guide and source images to be the same size
            # Ensure ai_l_uint8 has been resized
            if original_l_uint8.shape != ai_l_uint8.shape:
                 ai_l_uint8 = cv2.resize(ai_l_uint8, (original_l_uint8.shape[1], original_l_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Create and apply the guided filter
            guided_filter = cv2.ximgproc.createGuidedFilter(guide=ai_l_uint8, radius=radius, eps=eps)
            cleaned_l = guided_filter.filter(src=original_l_uint8)
        
            return np.clip(cleaned_l, 0, 255).astype(np.uint8)

    def _create_color_mask(self, model_output_lab_uint8: np.ndarray, threshold: float = 10.0, blur_kernel_size: int = 11) -> np.ndarray:
            """
            Creates a mask based on the a*b* channels of the model output to protect low-saturation areas.

            Args:
                model_output_lab_uint8 (np.ndarray): The model's output LAB image (uint8).
                threshold (float): The maximum color distance to be considered "colorless".
                blur_kernel_size (int): The kernel size for Gaussian blur to feather the mask edges.

            Returns:
                np.ndarray: A float32 mask with a value range of [0, 1]. 1.0 represents colorless areas.
            """
            a_channel = model_output_lab_uint8[:, :, 1].astype(np.float32)
            b_channel = model_output_lab_uint8[:, :, 2].astype(np.float32)

            # Calculate the Euclidean distance of each pixel's a*b* value from neutral gray (128, 128)
            # This distance represents color saturation
            color_distance = np.sqrt((a_channel - 128)**2 + (b_channel - 128)**2)

            # Normalize the distance to a [0, 1] range, where 0 is colorless and 1 is saturated
            # We use a threshold to define what distance is considered "saturated"
            mask = 1.0 - np.clip(color_distance / threshold, 0, 1.0)
        
            # Blur the mask to create a smooth transition
            if blur_kernel_size > 0:
                if blur_kernel_size % 2 == 0:
                    blur_kernel_size += 1
                mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

            return mask

    def _create_text_mask(self, original_l_channel: np.ndarray, dilation_kernel_size: int = 3, blur_kernel_size: int = 21) -> np.ndarray:
        """
        Creates a feathered mask to identify text areas from the original L channel.

        Args:
            original_l_channel (np.ndarray): The original luminance channel (uint8).
            dilation_kernel_size (int): The kernel size for morphological dilation to connect text.
            blur_kernel_size (int): The kernel size for Gaussian blur to feather the mask edges.

        Returns:
            np.ndarray: A float32 mask of the same size, with values [0, 1]. 
                        1.0 indicates a text area, 0.0 indicates a non-text area.
        """
        # 1. Use Otsu's thresholding to isolate black text.
        # THRESH_BINARY_INV makes the black text white (255) in the mask.
        _, text_mask = cv2.threshold(original_l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Dilate the mask to connect nearby characters and make the text blocks more solid.
        if dilation_kernel_size > 0:
            kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        # 3. Apply Gaussian blur to create soft/feathered edges for smooth blending.
        # The kernel size must be an odd number.
        if blur_kernel_size > 0:
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            text_mask = cv2.GaussianBlur(text_mask, (blur_kernel_size, blur_kernel_size), 0)

        # 4. Normalize the mask to be in the range [0.0, 1.0] for blending.
        return text_mask.astype(np.float32) / 255.0

    def _sharpen_l_channel(self, l_channel_uint8: np.ndarray, amount: float = 0.5, kernel_size: int = 5) -> np.ndarray:
            """
            Applies unsharp masking to sharpen a uint8 L channel.

            Args:
                l_channel_uint8 (np.ndarray): The L channel to be sharpened.
                amount (float): The sharpening strength; 0 means no change.
                kernel_size (int): The kernel size for Gaussian blur; must be an odd number.

            Returns:
                np.ndarray: The sharpened L channel (uint8).
            """
            if amount <= 0:
                return l_channel_uint8
        
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create a blurred version of the L channel
            blurred = cv2.GaussianBlur(l_channel_uint8, (kernel_size, kernel_size), 0)
        
            # Efficiently calculate the sharpened channel using addWeighted
            # Formula: sharpened = original * (1 + amount) + blurred * (-amount)
            sharpened = cv2.addWeighted(l_channel_uint8, 1.0 + amount, blurred, -amount, 0)
        
            return sharpened

    def colorize(self, use_original_l: bool = True, descreen_mode: str = 'auto',
                     post_processing_descreen_threshold: float = 1.5,
                     descreen_method: str = 'guided', 
                     descreen_strength: float = 0.8, guided_filter_radius: int = 4, guided_filter_eps: float = 200.0,
                     protect_text_via_mask: bool = True, masking_method: str = 'color', 
                     descreen_sharpen_strength: float = 0.0):
            """
            Perform colorization.
        
            Args:
                use_original_l (bool): If True, replaces the luminance of the colorized output with the
                                       luminance of the original image to preserve details. Defaults to True.
                descreen_mode (str): The descreening mode. Options are 'auto', 'force_on', 'force_off'.
                                     'auto' will automatically detect if descreening is needed.
                descreen_method (str): The descreening method. Options are 'pyramid', 'guided'. Only effective when descreen_mode is not 'force_off'.
                descreen_strength (float): The blend strength for the 'pyramid' method. A higher value uses more of the model's details. Defaults to 0.8.
                guided_filter_radius (int): The filter radius for the 'guided' method.
                guided_filter_eps (float): The smoothing strength for the 'guided' method.
                protect_text_via_mask (bool): Whether to enable a mask to protect details.
                masking_method (str): The method for generating the mask. Options are 'text' (based on luminance) or 'color' (based on color saturation).
                descreen_sharpen_strength (float): The sharpening strength when applying global descreening. This is applied to the L channel. Set to 0 to disable. Recommended values are 0.3-0.5.
            """
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

                final_l = original_l # Default to original L
                l_descreened = None

                # Automatic detection and mode selection logic
                apply_descreening = False
                if descreen_mode == 'force_on':
                    apply_descreening = True
                elif descreen_mode == 'auto':
                    # Directly read the stored state, no need to detect again
                    if self.scan_artifact_score > post_processing_descreen_threshold:
                        print(f"[Info] Post-processing descreen triggered (Score {self.scan_artifact_score:.2f} > Loose Threshold {post_processing_descreen_threshold}).")
                        apply_descreening = True
                    else:
                        print(f"[Info] Post-processing descreen skipped (Score {self.scan_artifact_score:.2f} <= Loose Threshold {post_processing_descreen_threshold}).")

                if apply_descreening:
                    # Descreening method selection logic
                    if descreen_method == 'pyramid':
                        l_descreened = self._merge_with_model_l_pyramid(original_l, model_output_rgb_uint8, levels=3, blend=descreen_strength)
                    elif descreen_method == 'guided':
                        # Generate the AI L channel as a guide
                        ai_l_guide = cv2.resize(model_output_lab[:, :, 0], (original_l.shape[1], original_l.shape[0]), interpolation=cv2.INTER_CUBIC)
                        l_descreened = self._clean_l_with_ai_guidance(original_l, ai_l_guide, radius=guided_filter_radius, eps=guided_filter_eps)

                if l_descreened is not None:
                    if protect_text_via_mask:
                        mask = None
                        target_shape = (original_l.shape[1], original_l.shape[0]) # (width, height)

                        # ---- [Modified Mask Selection Logic] ----
                        if masking_method == 'text':
                            # Create a mask where text is close to 1.0 and background is close to 0.0
                            mask = self._create_text_mask(original_l)
                        elif masking_method == 'color':
                            # Create a mask from the low-resolution model output
                            low_res_mask = self._create_color_mask(model_output_lab)
                            # Resize the mask to the same size as the original L channel
                            mask = cv2.resize(low_res_mask, target_shape, interpolation=cv2.INTER_CUBIC)

                        if mask is not None:
                            # For blending, we need to convert L channels to float
                            original_l_float = original_l.astype(np.float32)
                            l_descreened_float = l_descreened.astype(np.float32)
                        
                            # Blend the two L channels.
                            # Where mask is 1 (text), we use original_l.
                            # Where mask is 0 (no text), we use l_pyramid_fused.
                            final_l_float = (original_l_float * mask) + (l_descreened_float * (1.0 - mask))
                            final_l = np.clip(final_l_float, 0, 255).astype(np.uint8)
                        else:
                            final_l = l_descreened
                    else:
                        # No protection, apply pyramid merge globally
                        final_l = l_descreened

                    # If the user has specified a sharpening strength, perform sharpening
                    if descreen_sharpen_strength > 0:
                        final_l = self._sharpen_l_channel(final_l, amount=descreen_sharpen_strength)

                # 3. Resize the 'a' and 'b' channels from the model output to match the proc_l image's size
                target_h, target_w = final_l.shape
                resized_a = cv2.resize(model_output_lab[:, :, 1], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                resized_b = cv2.resize(model_output_lab[:, :, 2], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
                # 4. Merge the processed L channel with the resized a and b channels
                final_lab = np.stack([final_l, resized_a, resized_b], axis=-1).astype(np.uint8)

                # 5. Convert the final LAB image back to RGB and scale to [0, 1] float
                final_rgb_uint8 = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)

                return final_rgb_uint8.astype(np.float32) / 255.0
            else:
                # If not using original L, resize the model's direct output
                # to the original image's resolution using a smarter method.
                if self.original_image is None:
                    raise RuntimeError("The original image is not set. Please call 'set_image' before 'colorize'.")

                # Get original dimensions
                original_h, original_w = self.original_image.shape[:2]
            
                # --- Smarter Resize Method ---
            
                # 1. The 'result' is float32 [0.0, 1.0] RGB. Convert to uint8 [0-255] for color space conversion.
                result_uint8 = (result * 255).astype(np.uint8)

                # 2. Convert from RGB to LAB color space.
                # The L channel holds luminance (details), a and b hold color.
                lab_image = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2LAB)

                # 3. Split the LAB channels.
                l_channel, a_channel, b_channel = cv2.split(lab_image)
            
                # 4. Resize each channel individually.
                # We use INTER_CUBIC for high quality on all channels.
                # You could even experiment with INTER_LANCZOS4 for the L channel.
                resized_l = cv2.resize(l_channel, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
                resized_a = cv2.resize(a_channel, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
                resized_b = cv2.resize(b_channel, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
            
                # 5. Merge the resized channels back into a single LAB image.
                merged_lab = cv2.merge([resized_l, resized_a, resized_b])

                # 6. Convert the LAB image back to RGB.
                final_rgb_uint8 = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
            
                # 7. Convert back to float32 [0.0, 1.0] to match the function's expected output format.
                resized_result = final_rgb_uint8.astype(np.float32) / 255.0

                return resized_result
