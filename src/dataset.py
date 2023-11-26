import json
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import Compose

from src.yolo_helper import coco_to_yolo

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

def get_pandas_dataset(dataset_path: str, val: bool = False) -> tuple[pd.DataFrame, dict[str, list[dict[str, str]]]]:
    data_json = json.load(open(dataset_path))
    data = pd.DataFrame()

    if not val:
        data = pd.DataFrame(data_json['annotations'])
        classes = data_json['categories']

        data['bbox'] = [coco_to_yolo(i[0], i[1], i[2], i[3], IMAGE_HEIGHT, IMAGE_WIDTH) for i in data.bbox]
        data = data.groupby('image_id').agg({'category_id': list, 'bbox': list}).reset_index()

        images = dict({i['id']: i['file_name'] for i in data_json['images']})
        data['file_name'] = [images[i] for i in data.image_id]
        return data, classes
    
    data['file_name'] = [i['file_name'] for i in data_json['images']]
    data['id'] = [i['id'] for i in data_json['images']]
    return data, {}


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: pd.DataFrame, imageBasePath: str, val: bool = False, transforms: Optional[Compose] = None) -> None:
        self._dataset = dataset
        self._transforms = transforms
        self._imageBasePath = imageBasePath
        self._val = val

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_name = self._dataset['file_name'][idx]
        image = Image.open(self._imageBasePath + image_name).convert('RGB')
        
        target = {}
        if not self._val:
            target['boxes'] = torch.as_tensor(self._dataset['bbox'][idx])
            target['labels'] = torch.as_tensor(self._dataset['category_id'][idx])
        
        return T.ToTensor()(image), target

    def __len__(self) -> int:
        return len(self._dataset)


def split_dataset(dataset: FrameDataset) -> tuple[FrameDataset, FrameDataset]:
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    return train_dataset, test_dataset
