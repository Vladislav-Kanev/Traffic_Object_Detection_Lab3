import numpy as np
import albumentations as A
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.augmentations.transforms import ColorJitter 
# from torchvision.transforms import v2 as T
from albumentations.pytorch.transforms import ToTensorV2

def get_transform(train):
    transforms = []
    if train:
        transforms = [
        A.HorizontalFlip(p=0.3),
        ColorJitter(p=0.3),
        Affine(scale = (0.8, 0.8), p=0.4),
        ]
    # transforms.append(A.ToDtype(torch.float, scale=True))
    # transforms.append(A.ToPureTensor())
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    # transformed = transform(image=np.array(image) , bboxes=bboxes, category_ids=category_ids)

    # category_ids_to_name = dict((i['id'], i['name']) for i in classes)
    # visualize(
    #     transformed['image'],
    #     transformed['bboxes'],
    #     transformed['category_ids'],
    #     category_ids_to_name,
    # )