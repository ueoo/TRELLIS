from typing import *

import torch
import torch.nn as nn

from .lora_mixin import LoRAMixin
from .sparse_structure_flow import (
    SparseStructureFlowModel,
    SparseStructureLatentCondSparseStructureFlowModel,
)


class LoRASparseStructureFlowModel(LoRAMixin, SparseStructureFlowModel):
    pass


class LoRASparseStructureLatentCondSparseStructureFlowModel(
    LoRAMixin, SparseStructureLatentCondSparseStructureFlowModel
):
    pass
