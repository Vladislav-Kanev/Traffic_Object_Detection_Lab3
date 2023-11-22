import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.optim import LRScheduler, Optimizer
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.model import FasterRCNN


class Trainer:
    def __init__(self, model: FasterRCNN, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 optimizer: Optimizer, scheduler: LRScheduler, num_epochs: int, device: torch.device,
                 autocast: bool) -> None:
        self._model = model
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
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
            for data in tqdm(self._train_dataloader, desc=f'Epoch {epoch}', total=len(self._train_dataloader)):
                with autocast(self._autocast):
                    imgs = []
                    targets = []
                    for d in data:
                        imgs.append(d[0].to(self._device))
                        targ = {}
                        targ['boxes'] = d[1]['boxes'].to(self._device)
                        targ['labels'] = d[1]['labels'].to(self._device)
                        targets.append(targ)
                    loss_dict = self._model(imgs, targets)
                loss = sum(v for v in loss_dict.values())

                iteration_loss = loss.cpu().detach().numpy()
                epoch_loss.append(iteration_loss)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._scheduler.step()
            val_map = self._test()
            print(f'Epoch {epoch}: train_loss {np.mean(epoch_loss)}, mAP {val_map}')

    @torch.inference_mode()
    def _test(self) -> float:
        self._model.eval()
        for data in self._test_dataloader:
            imgs = []
            targets = []
            for d in data:
                imgs.append(d[0].to(self._device))
                targ = {}
                targ['boxes'] = d[1]['boxes'].to(self._device)
                targ['labels'] = d[1]['labels'].to(self._device)
                targets.append(targ)

            loss_list = self._model(imgs, targets)
            self._metric.update(loss_list, targets)
        return self._metric.compute()

    def test(self) -> None:
        val_map = self._test()
        print(f'Validation: mAP {val_map}')
