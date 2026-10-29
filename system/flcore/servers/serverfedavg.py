"""FedAvg on images: the sequential fine-tuning baseline, and its joint-training upper bound.

Both train the network of the model section (from scratch, or a pretrained ViT with trainable head,
last_k, lora, prompt or full) with FedAvg: in every round the selected clients start from the global
state, train local_epochs epochs on their own images, and the server averages the returned states
weighted by training samples. Nothing protects old classes, so FedAvg forgets them (the lower
reference of class-incremental learning); FedAvgJoint trains on every task's data and is an upper bound.
Centralised training is the special case of one client: partition iid with num_clients 1.
"""
from __future__ import annotations

from flcore.clients.clientfedavg import clientFedAvg, clientFedAvgJoint
from flcore.servers.serverimage import TRAIN_HPARAMS, ImageServer
from utils.config_utils import HParam


class FedAvg(ImageServer):
    """FedAvg: FedAvg on the current task's images only (sequential fine-tuning, a REFERENCE baseline)."""

    CLIENT_CLASS = clientFedAvg
    RUN_FLAGS = ("REFERENCE",)
    HPARAMS = {
        **TRAIN_HPARAMS,
        "loss_classes": HParam("str", "seen", "classes in the cross-entropy: seen (every class so far) or task "
                                              "(the current task's classes only)", choices=("seen", "task")),
    }


class FedAvgJoint(ImageServer):
    """FedAvgJoint: FedAvg on the images of every task so far (joint training, an UPPER_BOUND)."""

    CLIENT_CLASS = clientFedAvgJoint
    RUN_FLAGS = ("UPPER_BOUND",)
    HPARAMS = dict(TRAIN_HPARAMS)
