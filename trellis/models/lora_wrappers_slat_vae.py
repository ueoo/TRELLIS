from typing import *

import torch

from .lora_mixin import LoRAMixin
from .structured_latent_vae import (
    ElasticSLatEncoder,
    ElasticSLatGaussianDecoder,
    SLatEncoder,
    SLatGaussianDecoder,
)


class LoRASLatEncoder(LoRAMixin, SLatEncoder):
    pass


class LoRAElasticSLatEncoder(LoRAMixin, ElasticSLatEncoder):
    pass


class LoRASLatGaussianDecoder(LoRAMixin, SLatGaussianDecoder):
    pass


class LoRAElasticSLatGaussianDecoder(LoRAMixin, ElasticSLatGaussianDecoder):
    pass
