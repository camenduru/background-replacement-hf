# Based on https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
import os
import requests

import cv2
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

upsampler = None


def init():
    global upsampler

    print("Initializing upscaler...")

    if not os.path.exists("weights"):
        os.mkdir("weights")
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        response = requests.get(url)
        with open('weights/RealESRGAN_x2plus.pth', 'wb') as f:
            f.write(response.content)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2, model_path="weights/RealESRGAN_x2plus.pth", model=model, device="cuda")


def upscale(image):
    original_numpy = np.array(image)
    original_opencv = cv2.cvtColor(original_numpy, cv2.COLOR_RGB2BGR)

    output, _ = upsampler.enhance(original_opencv, outscale=2)
    upscaled = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    return upscaled
