import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from src.model import FasterRCNN


class Trainer:
    def __init__(self, model: FasterRCNN, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 optimizer: Optimizer, num_epochs: int, device: torch.device) -> None:
        self._model = model
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._optimizer = optimizer
        self._num_epochs = num_epochs
        self._device = device
        self._metric = MeanAveragePrecision(iou_type="bbox")

    @property
    def model(self) -> FasterRCNN:
        return self._model

    def train(self) -> None:
        self._model.train()
        for epoch in range(self._num_epochs):
            epoch_loss = []
            for data in self._train_dataloader:
                imgs = []
                targets = []
                for d in data:
                    imgs.append(d[0].to(self._device))
                    targ = {}
                    targ['boxes'] = d[1]['boxes'].to(self._device)
                    targ['labels'] = d[1]['labels'].to(self._device)
                    targets.append(targ)
                with autocast():
                    loss_dict = self._model(imgs, targets)
                loss = sum(v for v in loss_dict.values())

                iteration_loss = loss.cpu().detach().numpy()
                epoch_loss.append(iteration_loss)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                print(f'Iteration: train_loss: {iteration_loss}')
            print(f'Epoch {epoch}: train_loss: {np.mean(epoch_loss)}')

    @torch.inference_mode()
    def test(self):
        self._model.eval()

        val_losses = []
        for data in self._test_dataloader:
            imgs = []
            targets = []
            for d in data:
                imgs.append(d[0].to(self._device))
                targ = {}
                targ['boxes'] = d[1]['boxes'].to(self._device)
                targ['labels'] = d[1]['labels'].to(self._device)
                targets.append(targ)

            loss_dict = self._model(imgs, targets)
            loss = sum(v for v in loss_dict.values()).cpu().detach().numpy()
            val_losses.append(loss)

        print(f'Validation: val_loss {np.mean(val_losses)}')
