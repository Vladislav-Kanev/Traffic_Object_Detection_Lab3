from typing import Union

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.model import FasterRCNN

SubmissionType = list[dict[str, Union[list[int, int, float, float], float, int]]]

class Trainer:
    def __init__(self,
                 model: FasterRCNN,
                 train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
                 optimizer: Optimizer, scheduler: LRScheduler, num_epochs: int, device: torch.device,
                 autocast: bool) -> None:
        self._model = model
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._val_dataloader = val_dataloader
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._num_epochs = num_epochs
        self._device = device
        self._metric = MeanAveragePrecision(iou_type="bbox")
        self._autocast = autocast

    @property
    def model(self) -> FasterRCNN:
        return self._model

    def train(self) -> None:
        
        for epoch in range(self._num_epochs):
            self._model.train()
            epoch_loss = []
            for data in tqdm(self._train_dataloader, desc=f'Epoch {epoch}'):
                with autocast(self._autocast):
                    imgs, targets = self._prepare_batch(data)
                    loss_dict = self._model(imgs, targets)
                loss = sum(v for v in loss_dict.values())

                iteration_loss = loss.cpu().detach().numpy()
                epoch_loss.append(iteration_loss)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._scheduler.step()
            map_value = self._test()
            print(f'Epoch {epoch}: train_loss {np.mean(epoch_loss)}, mAP {map_value}')

    @torch.inference_mode()
    def _test(self) -> float:
        self._model.eval()
        for data in tqdm(self._test_dataloader, desc='Test'):
            imgs, targets = self._prepare_batch(data)
            loss_list = self._model(imgs, targets)
            self._metric.update(loss_list, targets)
        metric_result = self._metric.compute()
        map_value: torch.Tensor = metric_result['map'].item()
        return map_value

    def test(self) -> None:
        map_value = self._test()
        print(f'Validation: mAP {map_value}')

    def _prepare_batch(self, data: list[list[torch.Tensor]]
    ) -> tuple[list[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(self._device))
            targ = {}
            targ['boxes'] = d[1]['boxes'].to(self._device)
            targ['labels'] = d[1]['labels'].to(self._device)
            targets.append(targ)
        return imgs, targets

    @torch.inference_mode()
    def eval(self) -> SubmissionType:
        self._model.eval()
        result_list = []
        for data in tqdm(self._val_dataloader, desc='Validation'):
            print(len(data))
            for sample in data:
                result_list.extend(self._model(sample[0].to(self._device)))
        return result_list
