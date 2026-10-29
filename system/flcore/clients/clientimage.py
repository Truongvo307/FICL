"""Base client of image algorithms: local training of the shared network on the client's own images.

Every round the client loads the model it received into the one network object (flcore/models/network.py),
trains it for local_epochs epochs of minibatch steps on its training rows, and uploads the trained state:

    rows           training_rows(): the current task's rows (FedAvgJoint: every task so far)
    loss           cross-entropy over loss_classes() (the classes seen so far, or the current task's only):
                   the logits of every other class are set to -inf, so they get no probability and no
                   gradient (network.cross_entropy: logsumexp - gather, deterministic on CUDA)
    optimizer      hp optimizer, momentum, weight_decay; created anew every round (no state across rounds)
    learning rate  local_learning_rate, constant or cosine over the rounds of a task (hp lr_schedule)
    randomness     batch order and augmentation from (algorithm seed, client, task, round, epoch, row); torch's
                   generator seeded with (algorithm seed, client, task, round) for anything else

A subclass changes the rows, the classes or the loss by overriding training_rows(), loss_classes() or
loss(); the loop over epochs and batches stays here.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:            # settings checks, listings and dry runs work without torch
    torch = None

from flcore.clients.clientbase import Client

TRAIN_TAG = 0x494D47           # "IMG": separates these streams from other uses of the algorithm seed


class ImageClient(Client):
    def attach(self, bundle, device: str, rounds_per_task: int) -> None:
        """Called by the server after it built the network: the shared network, its device, rounds per task."""
        self.bundle, self.device, self.rounds_per_task = bundle, device, int(rounds_per_task)

    # ---- what to train on
    def training_rows(self) -> np.ndarray:
        return self.train_rows()

    def loss_classes(self) -> np.ndarray:
        if self.hp.get("loss_classes", "seen") == "task":
            return self.task_classes()
        return self.seen_classes()

    def loss(self, logits, y, x):
        """Loss of one batch. logits [b, num_classes] with -inf outside loss_classes(); y the labels."""
        from flcore.models.network import cross_entropy

        return cross_entropy(logits, y)

    def training_seed(self, t: int, r: int) -> int:
        return int(np.random.SeedSequence([self._algo_seed, TRAIN_TAG, self.id, int(t), int(r)]).generate_state(1)[0])

    # ---- hooks of the shared loop
    def on_begin_task(self, t: int) -> None:
        self._incoming, self._upload, self._report = None, None, {}

    def receive(self, t: int, r, payload: dict) -> None:
        if r is not None:
            self._incoming = payload["state"]

    def train_round(self, t: int, r: int) -> None:
        from flcore.models.network import lr_at, make_optimizer, seeded

        rows = self.training_rows()
        self._upload, self._report = None, {}
        if len(rows) == 0:
            return                                       # no data this task: nothing to train or send
        if self._incoming is None:
            raise RuntimeError(f"client {self.id}: no model was received before round {r} of task {t}")
        b, dev = self.bundle, self.device
        seen = self.seen_classes()
        b.load_state(self._incoming, seen, holder=("client", self.id))
        allowed = np.zeros(self.num_classes, dtype=bool)
        allowed[np.asarray(self.loss_classes(), dtype=np.int64)] = True
        labels = self.store.labels(rows)                 # the guard: training rows only
        if not allowed[labels].all():
            raise RuntimeError(f"client {self.id}: a training label is not among the classes of the loss")
        blocked = torch.as_tensor(~allowed, device=dev)
        lr = lr_at(self.learning_rate, self.hp["lr_schedule"], r, self.rounds_per_task)
        opt = make_optimizer(b.trainable_parameters(), self.hp["optimizer"], lr, self.hp["momentum"],
                             self.hp["weight_decay"])
        loss_sum, count = 0.0, 0
        with seeded(self.training_seed(t, r), dev):
            b.train_mode()
            for epoch in range(int(self.local_epochs)):
                loss_sum, count = 0.0, 0                 # the report is the last epoch's mean loss
                key = [self._algo_seed, TRAIN_TAG, self.id, int(t), int(r), int(epoch)]
                for x, y, _rows in self.store.train_batches(rows, int(self.batch_size), key):
                    x = x.to(dev, non_blocking=True)
                    y = y.to(dev, non_blocking=True)
                    logits = b.net(x).masked_fill(blocked, float("-inf"))
                    loss = self.loss(logits, y, x)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    loss_sum += float(loss.detach()) * int(y.shape[0])
                    count += int(y.shape[0])
        b.eval_mode()
        self._upload = {"state": b.get_state(seen), "n": int(len(rows))}
        self._report = {"train_loss": loss_sum / max(count, 1), "n": int(len(rows))}

    def upload_payload(self, t: int, r: int):
        return self._upload

    def round_report(self) -> dict:
        return self._report
