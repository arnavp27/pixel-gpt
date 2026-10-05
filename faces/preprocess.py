import numpy as np
from PIL import Image, ImageOps
import torch


def crop_image(image, zoom=1.0, x=0.5, y=0.5):
    if not 1 <= zoom <= 4 or not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError("Invalid crop")
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = max(1, int(min(width, height) / zoom))
    left, top = round((width - side) * x), round((height - side) * y)
    return image.crop((left, top, left + side, top + side))


def encode(image, size=32, levels=16):
    if size < 4 or size % 2 or not 2 <= levels <= 256:
        raise ValueError("Use an even image size and 2 to 256 gray levels")
    image = ImageOps.fit(ImageOps.exif_transpose(image).convert("L"), (size, size),
                        method=Image.Resampling.LANCZOS)
    pixels = np.rint(np.asarray(image, dtype=np.float32) * (levels - 1) / 255).astype(np.uint8)
    return torch.from_numpy(np.concatenate((pixels[:, :size // 2].ravel(),
                                           pixels[:, size // 2:].ravel())))


def decode(tokens, size=32, levels=16):
    pixels = tokens.detach().cpu().numpy()
    if pixels.size != size * size:
        raise ValueError("Token count does not match the image size")
    left, right = pixels.reshape(2, size, size // 2)
    image = np.concatenate((left, right), axis=1).astype(np.float32)
    return Image.fromarray(np.rint(image * 255 / (levels - 1)).clip(0, 255).astype(np.uint8))


def mirror(left, size=32):
    if left.shape[-1] != size * size // 2:
        raise ValueError("Expected the left half of an image")
    right = left.reshape(*left.shape[:-1], size, size // 2).flip(-1).flatten(-2)
    return torch.cat((left, right), dim=-1)


def inputs_for(targets, levels=16):
    start = torch.full((len(targets), 1), levels, dtype=torch.long, device=targets.device)
    return torch.cat((start, targets[:, :-1].long()), dim=1)
