import importlib


__attributes = {
    "SparseStructure": "sparse_structure",
    "SparseFeat2Render": "sparse_feat2render",
    "SLat2Render": "structured_latent2render",
    "Slat2RenderGeo": "structured_latent2render",
    "SparseStructureLatent": "sparse_structure_latent",
    "TextConditionedSparseStructureLatent": "sparse_structure_latent",
    "ImageConditionedSparseStructureLatent": "sparse_structure_latent",
    "ImageAllConditionedSparseStructureLatent": "sparse_structure_latent",
    "SparseStructureLatentConditionedSparseStructureLatent": "sparse_structure_latent",
    "PrevImageConditionedSparseStructureLatent": "sparse_structure_latent",
    "PrevImageCondConditionedSparseStructureLatent": "sparse_structure_latent",
    "SLat": "structured_latent",
    "TextConditionedSLat": "structured_latent",
    "ImageConditionedSLat": "structured_latent",
    "ImageAllConditionedSLat": "structured_latent",
    "SLatConditionedSLat": "structured_latent",
    "PrevImageConditionedSLat": "structured_latent",
    "PrevImageCondConditionedSLat": "structured_latent",
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


# For Pylance
if __name__ == "__main__":
    from .sparse_feat2render import SparseFeat2Render
    from .sparse_structure import SparseStructure
    from .sparse_structure_latent import (
        ImageConditionedSparseStructureLatent,
        ImageAllConditionedSparseStructureLatent,
        PrevImageCondConditionedSparseStructureLatent,
        PrevImageConditionedSparseStructureLatent,
        SparseStructureLatent,
        SparseStructureLatentConditionedSparseStructureLatent,
        TextConditionedSparseStructureLatent,
    )
    from .structured_latent import (
        ImageConditionedSLat,
        ImageAllConditionedSLat,
        PrevImageCondConditionedSLat,
        PrevImageConditionedSLat,
        SLat,
        SLatConditionedSLat,
        TextConditionedSLat,
    )
    from .structured_latent2render import SLat2Render, Slat2RenderGeo
