import json
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import Compose


def get_pandas_dataset(dataset_path: str) -> tuple[pd.DataFrame, dict[str, list[dict[str, str]]]]:
    data_json = json.load(open(dataset_path))
    data = pd.DataFrame(data_json['annotations'])
    classes = data_json['categories']
    
    images = dict({i['id']: i['file_name'] for i in data_json['images']})
    data['bbox'] = [[i[0], i[1], i[0] + i[2], i[1] + i[3]] for i in data.bbox]
    data = data.groupby('image_id').agg({'category_id': list, 'bbox': list}).reset_index()

    data['file_name'] = [images[i] for i in data.image_id]
    return data, classes


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: pd.DataFrame, imageBasePath: str, transforms: Optional[Compose] = None) -> None:
        self._dataset = dataset
        self._transforms = transforms
        self._imageBasePath = imageBasePath

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_name = self._dataset['file_name'][idx]
        image = Image.open(self._imageBasePath + image_name).convert('RGB')
        
        target = {}
        target['boxes'] = torch.as_tensor(self._dataset['bbox'][idx])
        target['labels'] = torch.as_tensor(self._dataset['category_id'][idx])
        
        return T.ToTensor()(image), target

    def __len__(self) -> int:
        return len(self._dataset)


def split_dataset(dataset: FrameDataset) -> tuple[FrameDataset, FrameDataset]:
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])
    return train_dataset, test_dataset
