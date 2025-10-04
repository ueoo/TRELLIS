import copy
import json
import os
import sys

from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pt_to_safetensors import convert_file


# Global defaults for repository layout
DATA_ROOT = "/viscam/projects/4d-state-machine/"
REFERENCE_FOLDER = "TRELLIS-image-large"
OUTPUTS_ROOT = os.path.join(DATA_ROOT, "TRELLIS_outputs")

# Fallback bases for extra LoRA keys to copy reference base weights
LORA_BASE_FALLBACK = {
    "slat_flow_model": "slat_flow_model",
    "slat_cond_slat_flow_model": "slat_flow_model",
    "sparse_structure_flow_model": "sparse_structure_flow_model",
    "sparse_structure_cond_sparse_structure_flow_model": "sparse_structure_flow_model",
}


def _copy_reference(reference_ckpts_path: str, ref_rel: str, dest_root: str, dst_stem: str):
    ref_sf = os.path.join(reference_ckpts_path, f"{ref_rel}.safetensors")
    ref_js = os.path.join(reference_ckpts_path, f"{ref_rel}.json")
    dst_sf = os.path.join(dest_root, f"{dst_stem}.safetensors")
    dst_js = os.path.join(dest_root, f"{dst_stem}.json")
    os.makedirs(os.path.dirname(dst_sf), exist_ok=True)
    os.system(f"cp {ref_sf} {dst_sf}")
    os.system(f"cp {ref_js} {dst_js}")


def build_folder(
    *,
    out_folder: str,
    pipeline_name: str,
    rename_ops: dict,
    model_name_to_ckpt: dict,
    lora_wrapper_name: dict | None = None,
):
    reference_path = os.path.join(DATA_ROOT, REFERENCE_FOLDER)
    reference_ckpts_path = os.path.join(reference_path, "ckpts")

    out_path = os.path.join(DATA_ROOT, out_folder)
    dest_root = os.path.join(out_path, "ckpts")
    os.makedirs(dest_root, exist_ok=True)

    pipeline_json = os.path.join(reference_path, "pipeline.json")
    with open(pipeline_json, "r") as f:
        pipeline = json.load(f)

    new_pipeline = copy.deepcopy(pipeline)
    new_pipeline["name"] = pipeline_name
    new_pipeline["args"]["models"] = {}

    reference_models = pipeline["args"]["models"]
    requested_keys = rename_ops.keys()

    for key in tqdm(requested_keys):
        op_or_name = rename_ops[key]

        # Simple COPY from reference
        if op_or_name == "copy":
            ref_path = reference_models[key]
            ref_stem = ref_path.replace("ckpts/", "")

            _copy_reference(reference_ckpts_path, ref_stem, dest_root, ref_stem)
            new_pipeline["args"]["models"][key] = f"ckpts/{ref_stem}"
            continue

        # From a run directory (op), require a checkpoint filename
        ckpt_filename = model_name_to_ckpt[key]
        if not ckpt_filename.endswith(".pt"):
            ckpt_filename += ".pt"

        src_pt = os.path.join(OUTPUTS_ROOT, op_or_name, "ckpts", ckpt_filename)
        dst_sf = os.path.join(dest_root, f"{op_or_name}.safetensors")
        convert_file(src_pt, dst_sf, copy_add_data=False)

        with open(os.path.join(OUTPUTS_ROOT, op_or_name, "config.json"), "r") as f:
            finetune_json = json.load(f)
        type_key = "denoiser" if "flow" in key else "decoder"
        model_def = finetune_json["models"][type_key]

        # LoRA: also copy base and swap name to wrapper
        if lora_wrapper_name is not None:
            assert key in lora_wrapper_name, f"Key '{key}' not in lora_wrapper_name."

            base_key = LORA_BASE_FALLBACK[key]
            base_ref_stem = reference_models[base_key].replace("ckpts/", "")

            ref_sf = os.path.join(reference_ckpts_path, f"{base_ref_stem}.safetensors")
            base_sf = os.path.join(dest_root, f"{op_or_name}.base.safetensors")
            os.system(f"cp {ref_sf} {base_sf}")
            model_def = {"name": lora_wrapper_name[key], "args": model_def["args"]}

        with open(os.path.join(dest_root, f"{op_or_name}.json"), "w") as f:
            json.dump(model_def, f, indent=4)

        new_pipeline["args"]["models"][key] = f"ckpts/{op_or_name}"

    with open(os.path.join(out_path, "pipeline.json"), "w") as f:
        json.dump(new_pipeline, f, indent=4)

    print(f"Created folder: {out_folder}")
