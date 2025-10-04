import importlib


__attributes = {
    "SparseStructureEncoder": "sparse_structure_vae",
    "SparseStructureDecoder": "sparse_structure_vae",
    "SparseStructureFlowModel": "sparse_structure_flow",
    "SparseStructureLatentCondSparseStructureFlowModel": "sparse_structure_flow",
    "SLatEncoder": "structured_latent_vae",
    "SLatGaussianDecoder": "structured_latent_vae",
    "SLatRadianceFieldDecoder": "structured_latent_vae",
    "SLatMeshDecoder": "structured_latent_vae",
    "ElasticSLatEncoder": "structured_latent_vae",
    "ElasticSLatGaussianDecoder": "structured_latent_vae",
    "ElasticSLatRadianceFieldDecoder": "structured_latent_vae",
    "ElasticSLatMeshDecoder": "structured_latent_vae",
    "SLatFlowModel": "structured_latent_flow",
    "SLatCondSLatFlowModel": "structured_latent_flow",
    "ElasticSLatFlowModel": "structured_latent_flow",
    "ElasticSLatCondSLatFlowModel": "structured_latent_flow",
    "ElasticPrevImageCondSLatFlowModel": "structured_latent_flow",
    "LoRASparseStructureFlowModel": "lora_wrappers_sparse",
    "LoRASparseStructureLatentCondSparseStructureFlowModel": "lora_wrappers_sparse",
    "LoRASLatFlowModel": "lora_wrappers_slat",
    "LoRAElasticSLatFlowModel": "lora_wrappers_slat",
    "LoRASLatCondSLatFlowModel": "lora_wrappers_slat",
    "LoRAElasticSLatCondSLatFlowModel": "lora_wrappers_slat",
    "LoRASLatEncoder": "lora_wrappers_slat_vae",
    "LoRAElasticSLatEncoder": "lora_wrappers_slat_vae",
    "LoRASLatGaussianDecoder": "lora_wrappers_slat_vae",
    "LoRAElasticSLatGaussianDecoder": "lora_wrappers_slat_vae",
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import json
    import os

    from safetensors.torch import load_file

    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        path_parts = path.split("/")
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
        model_name = "/".join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, "r") as f:
        config = json.load(f)
    model_name = config["name"]
    model_args = {**config.get("args", {}), **kwargs}

    model = __getattr__(model_name)(**model_args)

    # For LoRA and non-LoRA, load the provided checkpoint directly.
    # LoRA runs should save full model parameters for direct loading.
    model.load_state_dict(load_file(model_file))

    return model


# For Pylance
if __name__ == "__main__":
    from .sparse_structure_flow import (
        SparseStructureFlowModel,
        SparseStructureLatentCondSparseStructureFlowModel,
    )
    from .sparse_structure_vae import SparseStructureDecoder, SparseStructureEncoder
    from .structured_latent_flow import (
        ElasticPrevImageCondSLatFlowModel,
        ElasticSLatCondSLatFlowModel,
        ElasticSLatFlowModel,
        PrevImageCondSLatFlowModel,
        SLatCondSLatFlowModel,
        SLatFlowModel,
    )
    from .structured_latent_vae import (
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatMeshDecoder,
        ElasticSLatRadianceFieldDecoder,
        SLatEncoder,
        SLatGaussianDecoder,
        SLatMeshDecoder,
        SLatRadianceFieldDecoder,
    )
