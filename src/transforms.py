import numpy as np
import albumentations as A
# from torchvision.transforms import v2 as T

    # A.Compose(
    #     [A.HorizontalFlip(p=1)],
    #     bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    # )

def get_transform(train):
    transforms = []
    if train:
        transforms.append(A.HorizontalFlip(p=1))
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