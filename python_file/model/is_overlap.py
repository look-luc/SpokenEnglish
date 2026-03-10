import torch
import torch.nn as nn
import torch.nn.functional as F


class is_overlap_model(nn.Module):
    def __init__(self) -> None:
        super(is_overlap_model, self).__init__()
