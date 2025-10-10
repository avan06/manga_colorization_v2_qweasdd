# Version

This release refactors the [original codebase](https://github.com/qweasdd/manga-colorization-v2) by moving it under the `src/manga_color_v2` directory and introduces a `pyproject.toml` file to support installation via `pip`.

```bash
pip install git+https://github.com/avan06/manga_colorization_v2_qweasdd.git
```
## My environment for reference

- Python = 3.10.11
- Torch = 2.8.0
- Torchvision = 0.23.0
- Cuda = 12.8

# Usage

## Load Model

```python
from manga_color_v2.colorizator import MangaColorizator
_device = "cuda" if torch.cuda.is_available() else "cpu"
# networks_v2_path: the file path to the "networks generator.zip"
# denoiser_v2_dir: the directory containing the "net_rgb.pth" file
# target_resolution: The target resolution, e.g., 512.
manga_colorizator_v2 = MangaColorizator(_device, networks_v2_path, denoiser_v2_dir, target_resolution=512)
```

## Inference

```python
source_pil = Image.open(manga_file.name).convert("RGB")
source_np = np.array(source_pil)
# Set apply_denoise to True to enable denoising. denoise_sigma controls the strength of denoising.
manga_colorizator_v2.set_image(source_np, apply_denoise=True, denoise_sigma=25)
# use_original_l: Outputs images at the original resolution. Con: May slightly alter style and can reveal noise from the original image.
colorization_lowres_np = manga_colorizator_v2.colorize(use_original_l=True)
colorized_image_np = (colorization_lowres_np * 255).astype(np.uint8)
```


## Credits

*   **[Tag2Pix](https://github.com/blandocs/Tag2Pix/)**: A Generative Adversarial Network (GAN) based method that colors line art based on text tags.
*   **[AlacGAN](https://github.com/orashi/AlacGAN)**: A user-guided deep learning model for coloring anime line art.
*   **[manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)**: This project aims to provide a feature for automatically colorizing black and white manga using AI. It integrates components from the aforementioned projects to achieve its colorization capabilities. The `extractor.py` file, used for image feature extraction, originates from Tag2Pix, while the `models.py` file, which contains the neural network model architecture, is derived from AlacGAN.


___

##
##
## **UPD!!!** **A demo of Manga Colorization v2.5 is now available [link](https://mangacol.com). Feel free to check it out!**


# Automatic colorization

1. Download [generator](https://drive.google.com/file/d/1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t/view?usp=sharing) and [denoiser](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view?usp=sharing) weights. Put generator and extractor weights in `networks` and denoiser weights in `denoising/models`.
2. To colorize image or folder of images, use the following command:
```
$ python inference.py -p "path to file or folder"
```

| Original      | Colorization      |
|------------|-------------|
| <img src="figures/bw1.jpg" width="512"> | <img src="figures/color1.png" width="512"> |
| <img src="figures/bw2.jpg" width="512"> | <img src="figures/color2.png" width="512"> |
| <img src="figures/bw3.jpg" width="512"> | <img src="figures/color3.png" width="512"> |
| <img src="figures/bw4.jpg" width="512"> | <img src="figures/color4.png" width="512"> |
| <img src="figures/bw5.jpg" width="512"> | <img src="figures/color5.png" width="512"> |
| <img src="figures/bw6.jpg" width="512"> | <img src="figures/color6.png" width="512"> |
