"""Networks of the image path (torch). Imported only by image algorithms, inside init_algorithm().

    cnn.py         networks trained from scratch: tiny_cnn, resnet18_cifar, resnet32
    vit.py         pretrained ViTs as encoders (final-norm CLS token), LoRA and prompt tokens
    transforms.py  evaluation transforms and training augmentation
    network.py     build_network(): encoder + linear head, what is trainable, the messages (states)

The facts that settings checks need (names, sizes, which modes exist) are in utils/model_catalog.py,
which does not import torch.
"""
