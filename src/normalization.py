import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms as T
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

def get_gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype='float64')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

    return x / np.sum(x)


def local_contrast_normalization(image, radius=9):
    image_tensor = torch.Tensor(np.array(image))
    if radius % 2 == 0:
        radius += 1

    c, h, w = image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]

    gaussian_filter = torch.Tensor(get_gaussian_filter((1, c, radius, radius)))
    filtered_out = F.conv2d(image_tensor, gaussian_filter, padding=radius-1)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))

    centered_image = image_tensor - filtered_out[:, mid:-mid, mid:-mid]

    sum_sqr_image = F.conv2d(centered_image.pow(
        2), gaussian_filter, padding=radius-1)
    s_deviation = sum_sqr_image[:, mid:-mid, mid:-mid].sqrt()
    per_img_mean = s_deviation.mean()

    divisor = np.maximum(per_img_mean.numpy(), s_deviation.numpy())
    divisor = np.maximum(divisor, 1e-4)
    new_image = centered_image / torch.Tensor(divisor)
    return new_image


def local_response_norm(image, size=3, alpha=1e-4, beta=0.75, k=1):
    avg_pool2d = torch._C._nn.avg_pool2d
    avg_pool3d = torch._C._nn.avg_pool3d

    dim = image.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         image (got {} dimensions)'.format(dim))
    div = image.mul(image).unsqueeze(1)
    if dim == 3:
        div = F.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = image.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = F.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)

    return image / div


def get_local_contrast_normalization(radius):
    custom_transform = A.Lambda(image=lambda img: local_contrast_normalization(img, radius))
    
    transform = A.Compose([
        custom_transform,
    ], additional_targets={'radius': 'image'})
    
    return transform
