import torch
import torchvision
import cv2
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from sourcecode.configs.make_cfg import Struct
"""
- type: RandomRotation
      prob: 0.3
      angle: [-10, 10]
    - type: RandomRotation
      prob: 0.3
      angle: [90, 90]
    - type: RandomHorizontalFlip
      prob: 0.5
    - type: RandomVerticalFlip
      prob: 0.5
    - type: RandomScale
      scale: [0.9, 1.1]
      prob: 0.7
    - type: RandomCrop
      output_size: [512, 512]
    - type: ColorJitter
      brightness: 0.4
      contrast: 0.4
      prob: 0.6
      saturation: 0.0
"""
def custom_transform(cfg, image):
    cfg = Struct(**cfg)
    if cfg.type == 'Rescale':
        return Rescale(cfg, image)
    elif cfg.type == 'RandomRotation':
        return RandomRotation(cfg, image)
    elif cfg.type == 'RandomHorizontalFlip':
        return RandomHorizontalFlip(cfg, image)
    elif cfg.type == 'RandomVerticalFlip':
        return RandomVerticalFlip(cfg, image)
    elif cfg.type == 'RandomScale':
        return RandomScale(cfg, image)
    elif cfg.type == 'RandomCrop':
        return RandomCrop(cfg, image)
    elif cfg.type == 'ColorJitter':
        return ColorJitter(cfg, image)
    elif cfg.type == 'GaussianBlur':
        return GaussianBlur(cfg, image)
    else:
        return image

def Rescale(cfg, img):
    output_size = cfg.output_size
    h, w = img.shape[:2]
    if output_size == (w, h):
        return img

    h_rate = output_size[1] / h
    w_rate = output_size[0] / w
    min_rate = min(h_rate, w_rate)
    new_h = int(h * min_rate)
    new_w = int(w * min_rate)

    img = cv2.resize(
        img, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    top = random.randint(0, output_size[1] - new_h)
    bottom = output_size[1] - new_h - top
    left = random.randint(0, output_size[0] - new_w)
    right = output_size[0] - new_w - left
    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255])

    return img

def RandomRotation(cfg, img):
    if random.random() < cfg.prob:
        angle = cfg.angle[0] + (cfg.angle[1] -
                                     cfg.angle[0]) * random.random()
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(
            img,
            matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[255, 255, 255])
        
    return img
def RandomHorizontalFlip(cfg, img):
    if random.random() < cfg.prob:
        img = cv2.flip(img, 1)
        
    return img

def RandomVerticalFlip(cfg, img):
    if random.random() < cfg.prob:
        img = cv2.flip(img, 0)
        
    return img

def GaussianBlur(cfg, img):
    if random.random() < cfg.prob:
        kernel_size = cfg.kernel_range[np.random.choice(len(cfg.kernel_range), 1)[0]]
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
       
    return img

def RandomCrop(cfg, img):
    h, w = img.shape[:2]
    crop_w, crop_h = cfg.output_size
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(
            img,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255])

    margin_h = max(img.shape[0] - cfg.output_size[1], 0)
    margin_w = max(img.shape[1] - cfg.output_size[0], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + cfg.output_size[1]
    crop_x1, crop_x2 = offset_w, offset_w + cfg.output_size[0]

    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    
    return img

def ColorJitter(cfg, img):
    img = Image.fromarray(np.array(img, dtype=np.uint8))
    if random.random() < cfg.prob:
        img = transforms.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation)(
                img)
    img = np.array(img)
    return img

def RandomScale(cfg, img):
    if random.random() < cfg.prob:
        if random.random() < cfg.hw_prob:
            scale_factor_x = cfg.scale[0] + (cfg.scale[1] -
                                        cfg.scale[0]) * random.random()
            scale_factor_y = cfg.scale[0] + (cfg.scale[1] -
                                        cfg.scale[0]) * random.random()
        else:
            scale_factor_x = cfg.scale[0] + (cfg.scale[1] -
                                        cfg.scale[0]) * random.random()
            scale_factor_y = scale_factor_x

        img = cv2.resize(
            img,
            None,
            fx=scale_factor_x,
            fy=scale_factor_y,
            interpolation=cv2.INTER_LINEAR)
    
    return img