import numpy as np
from torchvision import transforms
from PIL import Image
from torch.nn import init
import torch
import os

def save_concatenated_image(filepath, images):
    w, h = images[0][0].width, images[0][0].height
    concat = Image.new('RGB', (w * len(images[0]), h * len(images)))
    for i, row in enumerate(images):
        for j, img in enumerate(row):
            concat.paste(img, (j*w, i*h))
    concat.save(filepath, 'PNG')

def tensor2numpy(tensor):
    return tensor.data.cpu().numpy()

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_psnr_np(img1, img2):
    SE_map = (1.*img1-img2)**2
    cur_MSE = np.mean(SE_map)
    return 20*np.log10(255./np.sqrt(cur_MSE))

def calculate_psnr_torch(img1, img2):
    """
    img1, img2 expected to float RGB torch tensors, which are scaled to 0~1
    """
    img1 = (torch.clamp(img1.permute(0, 2, 3, 1), 0, 1) * 255.0).type(torch.uint8)
    img2 = (torch.clamp(img2.permute(0, 2, 3, 1), 0, 1) * 255.0).type(torch.uint8)

    SE_map = (1.*img1-img2)**2
    cur_MSE = torch.mean(SE_map.reshape(SE_map.shape[0], -1), axis=1)
    return 20*torch.log10(255./torch.sqrt(cur_MSE))

def rgb2image(rgb_tensor):
    # batch_size must be 1
    arr = tensor2numpy(rgb_tensor[0])
    arr = np.clip(np.transpose(arr, (1, 2, 0)), 0, 1) * 255.0
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, 'RGB')
    return img


"""
credit
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L539
"""
def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
