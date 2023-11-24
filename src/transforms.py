import albumentations as A
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.augmentations.transforms import ColorJitter


def get_transform(val: bool = False) -> A.Compose:
    transforms = []
    if not val:
        transforms = [
            A.HorizontalFlip(p=0.3),
            ColorJitter(p=0.3),
            Affine(scale = (0.8, 0.8), p=0.4),
        ]
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
