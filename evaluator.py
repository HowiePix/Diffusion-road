import torch
import torch.nn as nn

from common.registry import registry

@registry.register_evaluator("fid")
class FID:
    def __init__(self, image_size):
        pass