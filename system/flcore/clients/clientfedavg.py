"""Clients of FedAvg and FedAvgJoint (image algorithms): see clientimage.py for the local training."""
from __future__ import annotations

import numpy as np

from flcore.clients.clientimage import ImageClient


class clientFedAvg(ImageClient):
    """Trains on the current task's images only (no replay)."""


class clientFedAvgJoint(ImageClient):
    """Trains on the images of every task so far: needs past_train_access (the UPPER_BOUND flag)."""

    def training_rows(self) -> np.ndarray:
        t = self.current_task
        past = [self.data.past_train_rows(i) for i in range(t)]
        return np.concatenate(past + [self.train_rows()]).astype(np.int64)

    def loss_classes(self) -> np.ndarray:
        return self.seen_classes()
