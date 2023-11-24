import json
from typing import Optional

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms as T
from albumentations import Compose

from .normalization import local_contrast_normalization, local_response_norm


def get_pandas_dataset(dataset_path: str, val: bool = False) -> tuple[pd.DataFrame, dict[str, list[dict[str, str]]]]:
    data_json = json.load(open(dataset_path))
    data = pd.DataFrame()

    if not val:
        data = pd.DataFrame(data_json['annotations'])
        classes = data_json['categories']

        data['bbox'] = [[i[0], i[1], i[0] + i[2], i[1] + i[3]]
                        for i in data.bbox]
        data = data.groupby('image_id').agg(
            {'category_id': list, 'bbox': list, 'area': list, 'iscrowd': list}).reset_index()

        images = dict({i['id']: i['file_name'] for i in data_json['images']})
        data['file_name'] = [images[i] for i in data.image_id]
        return data, classes

    data['file_name'] = [i['file_name'] for i in data_json['images']]
    data['id'] = [i['id'] for i in data_json['images']]
    return data, {}


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: pd.DataFrame, imageBasePath: str, val: bool = False, transforms: Optional[Compose] = None, normalization: str = None) -> None:
        self._dataset = dataset
        self._transforms = transforms
        self._imageBasePath = imageBasePath
        self._val = val
        self._normalization = normalization

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_name = self._dataset['file_name'][idx]
        # image = Image.open(self._imageBasePath + image_name)
        image = read_image(self._imageBasePath + image_name)

        target = {}
        if not self._val:
            target['boxes'] = torch.as_tensor(
                [[min(value, 512) for value in sublist] for sublist in self._dataset['bbox'][idx]])
            target['labels'] = torch.as_tensor(
                self._dataset['category_id'][idx])
            target['image_id'] = self._dataset['image_id'][idx]
            target['area'] = torch.as_tensor(self._dataset['area'][idx])
            target['iscrowd'] = torch.as_tensor(self._dataset['iscrowd'][idx])

        if self._normalization == 'local_contrast_normalization':
            image = local_contrast_normalization(image=image)
        elif self._normalization == 'local_response_norm':
            image = local_response_norm(image=image)

        if self._transforms is not None:
            # print("iteration started", target['image_id'])
            transformed_image = self._transforms(image=np.array(T.ToPILImage()(image)), bboxes=target['boxes'], category_ids=target['labels'])
            # print("iteration completed")
            image = T.ToTensor()(transformed_image['image'])
            target['boxes'] = torch.as_tensor(transformed_image['bboxes'])

        return image, target

    def __len__(self) -> int:
        return len(self._dataset)


def split_dataset(dataset: FrameDataset) -> tuple[FrameDataset, FrameDataset]:
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [
                                                                0.9, 0.1])
    return train_dataset, test_dataset
