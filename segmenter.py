# Based on https://github.com/xuebinqin/DIS/blob/main/Colab_Demo.ipynb
import os
from huggingface_hub import hf_hub_download

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = None
ISNetDIS = None
normalize = None
im_preprocess = None
hypar = None
net = None


def init():
    global device, ISNetDIS, normalize, im_preprocess, hypar, net

    print("Initializing segmenter...")

    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
        os.mkdir("git")
        os.system(
            "git clone https://github.com/xuebinqin/DIS git/xuebinqin/DIS")
        hf_hub_download(repo_id="NimaBoscarino/IS-Net_DIS-general-use",
                        filename="isnet-general-use.pth", local_dir="saved_models")
        os.system("rm -r git/xuebinqin/DIS/IS-Net/__pycache__")
        os.system(
            "mv git/xuebinqin/DIS/IS-Net/* .")

    import models
    import data_loader_cache

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ISNetDIS = models.ISNetDIS
    normalize = data_loader_cache.normalize
    im_preprocess = data_loader_cache.im_preprocess

    # Set Parameters
    hypar = {}  # paramters for inferencing

    # load trained weights from this path
    hypar["model_path"] = "./saved_models"
    # name of the to-be-loaded weights
    hypar["restore_model"] = "isnet-general-use.pth"
    # indicate if activate intermediate feature supervision
    hypar["interm_sup"] = False

    # choose floating point accuracy --
    # indicates "half" or "full" accuracy of float number
    hypar["model_digit"] = "full"
    hypar["seed"] = 0

    # cached input spatial resolution, can be configured into different size
    hypar["cache_size"] = [1024, 1024]

    # data augmentation parameters ---
    # mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hypar["input_size"] = [1024, 1024]
    # random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
    hypar["crop_size"] = [1024, 1024]

    hypar["model"] = ISNetDIS()

    # Build Model
    net = build_model(hypar, device)


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose(
    [GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_image(im_pil, hypar):
    im = np.array(im_pil)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    # make a batch of image, shape
    return transform(im).unsqueeze(0), shape.unsqueeze(0)


def build_model(hypar, device):
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if (hypar["model_digit"] == "half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if (hypar["restore_model"] != ""):
        net.load_state_dict(torch.load(
            hypar["model_path"]+"/"+hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()
    return net


def predict(net, inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if (hypar["model_digit"] == "full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(
        device)  # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0]  # list of 6 results

    # B x 1 x H x W    # we want the first one which is the most accurate prediction
    pred_val = ds_val[0][0, :, :, :]

    # recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(
        pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi)  # max = 1

    if device == 'cuda':
        torch.cuda.empty_cache()
    # it is the mask we need
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8)


def segment(image):
    image_tensor, orig_size = load_image(image, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)

    mask = Image.fromarray(mask).convert('L')
    im_rgb = image.convert("RGB")

    cropped = im_rgb.copy()
    cropped.putalpha(mask)

    return [cropped, mask]
