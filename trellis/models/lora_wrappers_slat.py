from typing import *

import torch
import torch.nn as nn

from .lora_mixin import LoRAMixin
from .structured_latent_flow import (
    ElasticSLatCondSLatFlowModel,
    ElasticSLatFlowModel,
    SLatCondSLatFlowModel,
    SLatFlowModel,
)


class LoRASLatFlowModel(LoRAMixin, SLatFlowModel):
    pass


class LoRAElasticSLatFlowModel(LoRAMixin, ElasticSLatFlowModel):
    pass


class LoRASLatCondSLatFlowModel(LoRAMixin, SLatCondSLatFlowModel):
    pass


class LoRAElasticSLatCondSLatFlowModel(LoRAMixin, ElasticSLatCondSLatFlowModel):
    pass
