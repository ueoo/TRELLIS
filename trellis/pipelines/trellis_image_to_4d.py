import os

from contextlib import contextmanager
from typing import *

import numpy as np
import rembg
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from tqdm import trange

from ..modules import sparse as sp
from . import samplers
from .base import Pipeline


class TrellisImageTo4DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-4D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo4DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo4DPipeline, TrellisImageTo4DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo4DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(
            **args["sparse_structure_sampler"]["args"]
        )
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(**args["slat_sampler"]["args"])
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]

        new_pipeline._init_image_cond_model(args["image_cond_model"])

        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == "RGBA":
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert("RGB")
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, "rembg_session", None) is None:
                self.rembg_session = rembg.new_session("u2net")
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models["image_cond_model"](image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        ss_cond: bool = False,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        if ss_cond:
            flow_model = self.models["sparse_structure_cond_sparse_structure_flow_model"]
        else:
            flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=False).samples

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        sparse_structure = decoder(z_s) > 0
        coords = torch.argwhere(sparse_structure)[:, [0, 2, 3, 4]].int()

        return z_s, coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if "mesh" in formats:
            ret["mesh"] = self.models["slat_decoder_mesh"](slat)
        if "gaussian" in formats:
            ret["gaussian"] = self.models["slat_decoder_gs"](slat)
        if "radiance_field" in formats:
            ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        slat_cond: bool = False,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        if slat_cond:
            flow_model = self.models["slat_cond_slat_flow_model"]
        else:
            flow_model = self.models["slat_flow_model"]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=False).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        num_frames: int = 120,
        scene_name: str = None,
        first_frame: int = 1,
        ss_latent_prev_folder: str = None,
        slat_latent_prev_folder: str = None,
    ) -> list[dict]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if num_samples > 1:
            num_samples = 1
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        # Ensure conditioning tensors have a batch dimension matching num_samples
        torch.manual_seed(seed)

        # print(f"Sampling frame 0 of {num_frames}")
        z_s, coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        results_list = []
        results_list.append(self.decode_slat(slat, formats))
        prev_z_s = z_s
        prev_slat = slat

        try:
            # Precompute normalization tensors for reuse across frames
            std = torch.tensor(self.slat_normalization["std"]).to(self.device)
            mean = torch.tensor(self.slat_normalization["mean"]).to(self.device)
            batch_size = num_samples
            for frame_idx in trange(first_frame + 1, num_frames + first_frame, desc="Sampling frames"):
                # print(f"Sampling frame {frame_idx} of {num_frames}")
                if ss_latent_prev_folder:
                    ss_latent_path = os.path.join(ss_latent_prev_folder, f"{scene_name}_{frame_idx:04d}.npz")
                    ss_latent = np.load(ss_latent_path)
                    ss_latent = torch.from_numpy(ss_latent["mean"]).float().to(self.device)
                    ss_latent = ss_latent.unsqueeze(0)
                    prev_z_s = ss_latent

                ss_cond = {
                    "cond": prev_z_s,
                    "neg_cond": torch.zeros_like(prev_z_s),
                }

                if slat_latent_prev_folder:
                    slat_latent_path = os.path.join(slat_latent_prev_folder, f"{scene_name}_{frame_idx:04d}.npz")
                    data = np.load(slat_latent_path)
                    coords = torch.tensor(data["coords"]).int().to(self.device)
                    if coords.shape[1] == 3:
                        coords = torch.cat([torch.zeros(coords.shape[0], 1).int().to(self.device), coords], dim=1)
                    feats = torch.tensor(data["feats"]).float().to(self.device)
                    feats = (feats - mean) / std
                    prev_slat = sp.SparseTensor(coords=coords, feats=feats).to(self.device)

                else:
                    prev_slat_feats = prev_slat.feats
                    prev_slat_coords = prev_slat.coords
                    prev_slat_feats = (prev_slat_feats - mean) / std
                    prev_slat = sp.SparseTensor(coords=prev_slat_coords, feats=prev_slat_feats).to(self.device)

                slat_cond = {
                    "cond": prev_slat,
                    "neg_cond": sp.SparseTensor(coords=prev_slat.coords, feats=torch.zeros_like(prev_slat.feats)),
                }
                z_s, coords = self.sample_sparse_structure(
                    ss_cond, num_samples, sparse_structure_sampler_params, ss_cond=True
                )
                slat = self.sample_slat(slat_cond, coords, slat_sampler_params, slat_cond=True)
                results_list.append(self.decode_slat(slat, formats))
                prev_z_s = z_s
                prev_slat = slat

            # some sample goes wrong, return the results so far
            return results_list

        except Exception as e:
            print(f"Error sampling frames: {e}")
            return results_list

    @torch.no_grad()
    def run_latent_distance(
        self,
        image: Image.Image,
        scene_name: str,
        num_frames: int = 120,
        first_frame: int = 1,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        ss_latent_prev_folder: str = None,
        slat_latent_prev_folder: str = None,
        ss_latent_folder: str = None,
        slat_latent_folder: str = None,
    ) -> dict:
        """
        Generate structured latents and compute distances to ground-truth sparse structure and SLAT latents.

        Args:
            image (Image.Image): The image prompt.
            ss_latent_folder (str): Folder containing GT sparse structure latent as npz with key 'mean'.
            slat_latent_folder (str): Folder containing GT slat latents as npz with keys 'coords' and 'feats'.
            scene_name (str): Scene prefix used in GT filenames.
            num_frames (int): Number of frames to evaluate.
            first_frame (int): Index of the first frame (used for filename alignment).
            num_samples (int): Number of samples to generate (clamped to 1).
            seed (int): Random seed.
            sparse_structure_sampler_params (dict): Params for sparse structure sampler.
            slat_sampler_params (dict): Params for structured latent sampler.
            preprocess_image (bool): Whether to preprocess input image.
            ss_latent_prev_folder (str): Optional folder for previous-frame sparse structure latent conditioning.
            slat_latent_prev_folder (str): Optional folder for previous-frame slat conditioning.

        Returns:
            dict: Per-frame and overall latent distance metrics for sparse structure and SLAT.
        """
        if num_samples > 1:
            num_samples = 1
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)

        def _load_gt_ss_mean(frame_idx: int) -> torch.Tensor:
            gt_path = os.path.join(ss_latent_folder, f"{scene_name}_{frame_idx:04d}.npz")
            data = np.load(gt_path)
            if "mean" not in data:
                raise KeyError("GT sparse structure latent npz must contain 'mean'")
            gt_mean = torch.tensor(data["mean"]).float().to(self.device)
            return gt_mean

        # Helpers
        def _load_gt_slat(frame_idx: int) -> sp.SparseTensor:
            gt_path = os.path.join(slat_latent_folder, f"{scene_name}_{frame_idx:04d}.npz")
            data = np.load(gt_path)
            coords = torch.tensor(data["coords"]).int().to(self.device)
            if coords.shape[1] == 3:
                coords = torch.cat([torch.zeros(coords.shape[0], 1).int().to(self.device), coords], dim=1)
            feats = torch.tensor(data["feats"]).float().to(self.device)
            return sp.SparseTensor(coords=coords, feats=feats)

        def _compute_ss_latent_distances(
            g_zs: torch.Tensor, gt_mean: torch.Tensor
        ) -> Tuple[Optional[float], Optional[float]]:
            try:
                g_vec = g_zs.detach().float().view(-1)
                t_vec = gt_mean.detach().float().view(-1)
                if g_vec.numel() != t_vec.numel():
                    # size mismatch; cannot compute
                    return None, None
                l2 = float(F.mse_loss(g_vec, t_vec, reduction="mean").item())
                cos_sim = float(F.cosine_similarity(g_vec, t_vec, dim=0).item())
                cos_dist = 1.0 - cos_sim
                return l2, cos_dist
            except Exception:
                return None, None

        def _compute_slat_distances(
            g_slat: sp.SparseTensor, gt_slat: sp.SparseTensor
        ) -> Tuple[Optional[float], Optional[float], int, int, int]:
            gen_coords_np = g_slat.coords.detach().cpu().numpy()
            gt_coords_np = gt_slat.coords.detach().cpu().numpy()

            gen_index = {tuple(c.tolist()): i for i, c in enumerate(gen_coords_np)}
            gen_indices: list[int] = []
            gt_indices: list[int] = []
            for j, c in enumerate(gt_coords_np):
                key = tuple(c.tolist())
                i = gen_index.get(key)
                if i is not None:
                    gen_indices.append(i)
                    gt_indices.append(j)

            matched = len(gen_indices)
            if matched == 0:
                return None, None, int(g_slat.coords.shape[0]), int(gt_slat.coords.shape[0]), 0

            device = g_slat.feats.device
            gen_idx_t = torch.tensor(gen_indices, dtype=torch.long, device=device)
            gt_idx_t = torch.tensor(gt_indices, dtype=torch.long, device=device)
            gen_sel = g_slat.feats.index_select(0, gen_idx_t).float()
            gt_sel = gt_slat.feats.index_select(0, gt_idx_t).float()

            mean_mse = float(F.mse_loss(gen_sel, gt_sel, reduction="mean").item())
            cos_sim = float(F.cosine_similarity(gen_sel, gt_sel, dim=-1).mean().item())
            cos_dist = 1.0 - cos_sim
            return mean_mse, cos_dist, int(g_slat.coords.shape[0]), int(gt_slat.coords.shape[0]), matched

        # Initial frame generation (aligned to first_frame)
        z_s, coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)

        per_frame = []
        try:
            # SS latent metrics for first frame
            try:
                gt_ss_mean = _load_gt_ss_mean(first_frame)
                ss_lat_l2, ss_lat_cos = _compute_ss_latent_distances(z_s[0], gt_ss_mean)
            except Exception as e:
                print(f"Error computing SS latent distance for frame {first_frame}: {e}")
                ss_lat_l2 = ss_lat_cos = None

            gt_slat = _load_gt_slat(first_frame)
            m_l2, m_cos, n_gen, n_gt, n_match = _compute_slat_distances(slat, gt_slat)
            per_frame.append(
                {
                    "frame": int(first_frame),
                    # sparse structure latent metrics
                    "ss_latent_mean_l2": ss_lat_l2,
                    "ss_latent_mean_cosine_distance": ss_lat_cos,
                    # slat metrics
                    "slat_mean_l2": m_l2,
                    "slat_mean_cosine_distance": m_cos,
                    "slat_num_gen_points": int(n_gen),
                    "slat_num_gt_points": int(n_gt),
                    "slat_num_matched_points": int(n_match),
                    "slat_match_ratio_gen": float(n_match / n_gen) if n_gen > 0 else None,
                    "slat_match_ratio_gt": float(n_match / n_gt) if n_gt > 0 else None,
                }
            )
        except Exception as e:
            print(f"Error computing distance for frame {first_frame}: {e}")

        prev_z_s = z_s
        prev_slat = slat

        try:
            # Precompute normalization tensors reused across frames
            std = torch.tensor(self.slat_normalization["std"]).to(self.device)
            mean = torch.tensor(self.slat_normalization["mean"]).to(self.device)

            for frame_idx in trange(first_frame + 1, num_frames + first_frame, desc="Sampling frames for distance"):
                if ss_latent_prev_folder:
                    ss_latent_path = os.path.join(ss_latent_prev_folder, f"{scene_name}_{frame_idx:04d}.npz")
                    ss_latent = np.load(ss_latent_path)
                    ss_latent = torch.from_numpy(ss_latent["mean"]).float().to(self.device)
                    ss_latent = ss_latent.unsqueeze(0)
                    prev_z_s = ss_latent

                ss_cond = {
                    "cond": prev_z_s,
                    "neg_cond": torch.zeros_like(prev_z_s),
                }

                if slat_latent_prev_folder:
                    slat_latent_path = os.path.join(slat_latent_prev_folder, f"{scene_name}_{frame_idx:04d}.npz")
                    data = np.load(slat_latent_path)
                    coords = torch.tensor(data["coords"]).int().to(self.device)
                    if coords.shape[1] == 3:
                        coords = torch.cat([torch.zeros(coords.shape[0], 1).int().to(self.device), coords], dim=1)
                    feats = torch.tensor(data["feats"]).float().to(self.device)
                    feats = (feats - mean) / std
                    prev_slat = sp.SparseTensor(coords=coords, feats=feats).to(self.device)
                else:
                    prev_slat_feats = prev_slat.feats
                    prev_slat_coords = prev_slat.coords
                    prev_slat_feats = (prev_slat_feats - mean) / std
                    prev_slat = sp.SparseTensor(coords=prev_slat_coords, feats=prev_slat_feats).to(self.device)

                slat_cond = {
                    "cond": prev_slat,
                    "neg_cond": sp.SparseTensor(coords=prev_slat.coords, feats=torch.zeros_like(prev_slat.feats)),
                }
                z_s, coords = self.sample_sparse_structure(
                    ss_cond, num_samples, sparse_structure_sampler_params, ss_cond=True
                )
                slat = self.sample_slat(slat_cond, coords, slat_sampler_params, slat_cond=True)

                # Compute distance to GT for this frame
                try:
                    # SS latent metrics
                    try:
                        gt_ss_mean = _load_gt_ss_mean(frame_idx)
                        ss_lat_l2, ss_lat_cos = _compute_ss_latent_distances(z_s[0], gt_ss_mean)
                    except Exception as e:
                        print(f"Error computing SS latent distance for frame {frame_idx}: {e}")
                        ss_lat_l2 = ss_lat_cos = None

                    gt_slat = _load_gt_slat(frame_idx)
                    m_l2, m_cos, n_gen, n_gt, n_match = _compute_slat_distances(slat, gt_slat)
                    per_frame.append(
                        {
                            "frame": int(frame_idx),
                            # sparse structure latent metrics
                            "ss_latent_mean_l2": ss_lat_l2,
                            "ss_latent_mean_cosine_distance": ss_lat_cos,
                            # slat metrics
                            "slat_mean_l2": m_l2,
                            "slat_mean_cosine_distance": m_cos,
                            "slat_num_gen_points": int(n_gen),
                            "slat_num_gt_points": int(n_gt),
                            "slat_num_matched_points": int(n_match),
                            "slat_match_ratio_gen": float(n_match / n_gen) if n_gen > 0 else None,
                            "slat_match_ratio_gt": float(n_match / n_gt) if n_gt > 0 else None,
                        }
                    )
                except Exception as e:
                    print(f"Error computing distance for frame {frame_idx}: {e}")

                prev_z_s = z_s
                prev_slat = slat

        except Exception as e:
            print(f"Error during sampling for distance computation: {e}")

        # Aggregate overall metrics (only frames with valid values)
        ss_lat_l2_vals = [f["ss_latent_mean_l2"] for f in per_frame if f.get("ss_latent_mean_l2") is not None]
        ss_lat_cos_vals = [
            f["ss_latent_mean_cosine_distance"]
            for f in per_frame
            if f.get("ss_latent_mean_cosine_distance") is not None
        ]
        slat_l2_vals = [f["slat_mean_l2"] for f in per_frame if f.get("slat_mean_l2") is not None]
        slat_cos_vals = [
            f["slat_mean_cosine_distance"] for f in per_frame if f.get("slat_mean_cosine_distance") is not None
        ]

        overall = {
            "frames_evaluated": int(len(per_frame)),
            # Sparse Structure Latent
            "ss_mean_l2": float(np.mean(ss_lat_l2_vals)) if len(ss_lat_l2_vals) > 0 else None,
            "ss_mean_cosine_distance": float(np.mean(ss_lat_cos_vals)) if len(ss_lat_cos_vals) > 0 else None,
            # SLAT
            "slat_mean_l2": float(np.mean(slat_l2_vals)) if len(slat_l2_vals) > 0 else None,
            "slat_mean_cosine_distance": float(np.mean(slat_cos_vals)) if len(slat_cos_vals) > 0 else None,
        }

        return {"per_frame": per_frame, "overall": overall}

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f"_old_inference_model", sampler._inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m"
                )

            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx : cond_idx + 1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":
            from .samplers import FlowEulerSampler

            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i : i + 1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i : i + 1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f"_old_inference_model")

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get("steps")
        with self.inject_sampler_multi_image("sparse_structure_sampler", len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.inject_sampler_multi_image("slat_sampler", len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
