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
manga_colorizator_v2 = MangaColorizator(_device, networks_v2_path, denoiser_v2_dir)
```

## Inference

```python
source_pil = Image.open(manga_file.name).convert("RGB")
source_np = np.array(source_pil)
manga_colorizator_v2.set_image(source_np, size=576, apply_denoise=True, denoise_sigma=25)
colorization_lowres_np = manga_colorizator_v2.colorize()
colorized_image_np = (colorization_lowres_np * 255).astype(np.uint8)
```

___
___

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
