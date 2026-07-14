import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from transformers import AutoImageProcessor

from useful_stuff.general_utils import convert_dtype_by_name, get_device, print_wise
from useful_stuff.image_processing.computational_models import (
    get_relevant_output_layers,
    imgANN,
)
from useful_stuff.image_processing.utils import get_usual_transform
from useful_stuff.parallel.parallel_funcs import master_workers_queue, parallel_setup


MODEL_REPOS = {
    "dino_v3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ijepa_vith14_1k": "facebook/ijepa_vith14_1k",
}


@dataclass
class FeatureExtractionCfg:
    folder_name: str = "talia_20each_tizi"
    model_name: str = "dino_v3_l"
    pkg: str = "hf"
    img_size: int = 224
    batch_size: int = 8
    pooling: str = "mean"
    num_workers: int = 0
    dataset_path: str | None = None
    output_dir: str | None = None
    repo_url: str | None = None
    revision: str | None = None
    layers: list[str] | None = None
    dtype: str = "float32"
    weights_type: str = "DEFAULT"
    trust_remote_code: bool = False
    overwrite: bool = False
    prepare_only: bool = False
# EOC


class ProcessorTransform:
    """Apply a Hugging Face image processor to one PIL image."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]
    # EOF
# EOC


"""
parse_feature_extraction_args
Parse command-line overrides into the feature-extraction configuration.

OUTPUT:
    - cfg: FeatureExtractionCfg -> feature-extraction parameters
"""
def parse_feature_extraction_args() -> FeatureExtractionCfg:
    parser = argparse.ArgumentParser(
        description="Extract intermediate ANN features from an image dataset."
    )
    parser.add_argument("--folder_name", default=FeatureExtractionCfg.folder_name)
    parser.add_argument(
        "--model_name",
        default=FeatureExtractionCfg.model_name,
        help="Model architecture/label used for layer lookup and output filenames.",
    )
    parser.add_argument(
        "--pkg",
        choices=["hf", "timm", "torchvision"],
        default=FeatureExtractionCfg.pkg,
    )
    parser.add_argument("--img_size", type=int, default=FeatureExtractionCfg.img_size)
    parser.add_argument(
        "--batch_size", type=int, default=FeatureExtractionCfg.batch_size
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "sum", "max", "min", "all"],
        default=FeatureExtractionCfg.pooling,
    )
    parser.add_argument(
        "--num_workers", type=int, default=FeatureExtractionCfg.num_workers
    )
    parser.add_argument("--dataset_path", default=FeatureExtractionCfg.dataset_path)
    parser.add_argument("--output_dir", default=FeatureExtractionCfg.output_dir)
    parser.add_argument(
        "--repo_url",
        default=FeatureExtractionCfg.repo_url,
        help="Optional Hugging Face repository; overrides the model-name source.",
    )
    parser.add_argument("--revision", default=FeatureExtractionCfg.revision)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=FeatureExtractionCfg.layers,
        help="Module paths to extract; otherwise use the project's known layer list.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default=FeatureExtractionCfg.dtype,
    )
    parser.add_argument("--weights_type", default=FeatureExtractionCfg.weights_type)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    return FeatureExtractionCfg(**vars(parser.parse_args()))
# EOF


"""
resolve_dataset_path
Resolve an explicit dataset path or the standard LivingstoneLab stimulus path.

INPUT:
    - paths: dict[str, str] -> environment-specific project paths
    - cfg: FeatureExtractionCfg -> feature-extraction parameters

OUTPUT:
    - dataset_path: Path -> resolved ImageFolder root
"""
def resolve_dataset_path(paths: dict[str, str], cfg: FeatureExtractionCfg) -> Path:
    if cfg.dataset_path is not None:
        dataset_path = Path(cfg.dataset_path).expanduser()
    else:
        dataset_path = Path(paths["livingstone_lab"]) / "Stimuli" / cfg.folder_name
    # end if cfg.dataset_path is not None:
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    # end if not dataset_path.is_dir():
    return dataset_path
# EOF


"""
resolve_output_dir
Resolve the feature directory while preserving the project's standard location.

INPUT:
    - paths: dict[str, str] -> environment-specific project paths
    - cfg: FeatureExtractionCfg -> feature-extraction parameters

OUTPUT:
    - output_dir: Path -> directory in which layer NPZ files are stored
"""
def resolve_output_dir(paths: dict[str, str], cfg: FeatureExtractionCfg) -> Path:
    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir).expanduser()
    elif "data_path" in paths:
        output_dir = Path(paths["data_path"]) / "models"
    else:
        output_dir = Path(paths["livingstone_lab"]) / "tiziano" / "models"
    # end if cfg.output_dir is not None:
    return output_dir
# EOF


"""
resolve_model_source
Resolve a Hugging Face alias, explicit repository, or direct repository identifier.

INPUT:
    - cfg: FeatureExtractionCfg -> feature-extraction parameters

OUTPUT:
    - model_source: str -> model repository passed to Hugging Face
"""
def resolve_model_source(cfg: FeatureExtractionCfg) -> str:
    if cfg.repo_url is not None:
        return cfg.repo_url
    # end if cfg.repo_url is not None:
    return MODEL_REPOS.get(cfg.model_name, cfg.model_name)
# EOF


"""
feature_save_path
Build the activation filename used by the downstream metrics analyses.

INPUT:
    - output_dir: Path -> model activation directory
    - cfg: FeatureExtractionCfg -> feature-extraction parameters
    - layer_name: str -> hooked module path

OUTPUT:
    - save_path: Path -> output NPZ path
"""
def feature_save_path(
    output_dir: Path, cfg: FeatureExtractionCfg, layer_name: str
) -> Path:
    model_label = cfg.model_name.replace("/", "_")
    file_name = (
        f"{cfg.folder_name}_{model_label}_{cfg.img_size}_{layer_name}"
        f"_features_{cfg.pooling}pool.npz"
    )
    return output_dir / file_name
# EOF


"""
split_layers
Split model layers into non-empty contiguous extraction groups.

INPUT:
    - layer_names: list[str] -> ordered model layer names
    - n_groups: int -> maximum number of groups

OUTPUT:
    - layer_groups: list[list[str]] -> non-empty contiguous layer groups
"""
def split_layers(layer_names: list[str], n_groups: int) -> list[list[str]]:
    if n_groups < 1:
        raise ValueError("n_groups must be at least one.")
    # end if n_groups < 1:
    groups = np.array_split(np.asarray(layer_names, dtype=object), n_groups)
    return [group.tolist() for group in groups if len(group) > 0]
# EOF


"""
load_feature_extraction_inputs
Load the model wrapper and its ordered, preprocessed image dataset.

INPUT:
    - paths: dict[str, str] -> environment-specific project paths
    - cfg: FeatureExtractionCfg -> feature-extraction parameters
    - device: torch.device -> inference device

OUTPUT:
    - ann: imgANN -> loaded model wrapper
    - loader: DataLoader -> ordered image batches
    - model_input_key: str -> model forward argument containing image tensors
"""
def load_feature_extraction_inputs(
    paths: dict[str, str], cfg: FeatureExtractionCfg, device
):
    dtype = convert_dtype_by_name(cfg.dtype, "torch")
    repo_url = resolve_model_source(cfg) if cfg.pkg == "hf" else None
    ann = imgANN(
        model_name=cfg.model_name,
        pkg=cfg.pkg,
        img_size=cfg.img_size,
        relevant_layers=cfg.layers,
        pooling=cfg.pooling,
        weights_type=cfg.weights_type,
        dtype=dtype,
        repo_url=repo_url,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
        device=device,
    )

    if cfg.pkg == "hf":
        model_source = resolve_model_source(cfg)
        processor = AutoImageProcessor.from_pretrained(
            model_source,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=False,
        )
        transform = ProcessorTransform(processor)
        model_input_key = "pixel_values"
    else:
        transform = get_usual_transform(resize_size=cfg.img_size)
        model_input_key = "x"
    # end if cfg.pkg == "hf":

    dataset = ImageFolder(
        root=resolve_dataset_path(paths, cfg),
        transform=transform,
        is_valid_file=lambda path: Path(path).suffix.lower() in IMG_EXTENSIONS,
        allow_empty=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"No images found in {dataset.root}.")
    # end if len(dataset) == 0:
    sample_shape = dataset[0][0].shape[-2:]
    if sample_shape != (cfg.img_size, cfg.img_size):
        raise ValueError(
            f"Preprocessing returned {sample_shape}, expected "
            f"{(cfg.img_size, cfg.img_size)}."
        )
    # end if sample_shape != (cfg.img_size, cfg.img_size):
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    return ann, loader, model_input_key
# EOF


"""
extract_layer_group
Extract and save pooled activations for several layers in one dataset pass.

INPUT:
    - paths: dict[str, str] -> environment-specific project paths
    - rank: int -> MPI rank, or zero for serial extraction
    - layer_names: list[str] -> layers extracted in this dataset pass
    - ann: imgANN -> loaded model wrapper
    - loader: DataLoader -> ordered image batches
    - model_input_key: str -> model forward argument containing image tensors
    - cfg: FeatureExtractionCfg -> feature-extraction parameters

OUTPUT:
    - None -> one compressed NPZ file is saved per selected layer
"""
def extract_layer_group(
    paths,
    rank,
    layer_names,
    ann,
    loader,
    model_input_key,
    cfg,
):
    output_dir = resolve_output_dir(paths, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_to_extract = [
        layer_name
        for layer_name in layer_names
        if cfg.overwrite or not feature_save_path(output_dir, cfg, layer_name).exists()
    ]
    if not layers_to_extract:
        print_wise("all assigned layers already exist", rank=rank)
        return
    # end if not layers_to_extract:

    ann.create_forward_hook(layers_to_extract)
    layer_features = {layer_name: [] for layer_name in layers_to_extract}
    try:
        with torch.inference_mode():
            for batch_idx, (images, _) in enumerate(loader):
                model_inputs = {model_input_key: images.to(ann.device)}
                ann.extract_features(model_inputs)
                for layer_name in layers_to_extract:
                    features = ann.features[layer_name].detach().cpu().numpy()
                    layer_features[layer_name].append(features)
                # end for layer_name in layers_to_extract:
                print_wise(
                    f"processed batch {batch_idx + 1}/{len(loader)}", rank=rank
                )
            # end for batch_idx, (images, _) in enumerate(loader):
        # end with torch.inference_mode():
    finally:
        ann.clear_hooks()
    # end try:

    for layer_name in layers_to_extract:
        features = np.concatenate(layer_features[layer_name], axis=0).T
        save_path = feature_save_path(output_dir, cfg, layer_name)
        np.savez_compressed(save_path, features)
        print_wise(f"saved {features.shape} features at {save_path}", rank=rank)
    # end for layer_name in layers_to_extract:
# EOF


"""
run_feature_extraction
Prepare a Hugging Face model or extract layers serially or with MPI workers.

INPUT:
    - paths: dict[str, str] -> environment-specific project paths
    - cfg: FeatureExtractionCfg -> feature-extraction parameters

OUTPUT:
    - None
"""
def run_feature_extraction(paths: dict[str, str], cfg: FeatureExtractionCfg):
    if cfg.prepare_only:
        if cfg.pkg != "hf":
            raise ValueError("prepare_only is supported only for Hugging Face models.")
        # end if cfg.pkg != "hf":
        model_source = resolve_model_source(cfg)
        AutoImageProcessor.from_pretrained(
            model_source,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=False,
        )
        imgANN(
            model_name=cfg.model_name,
            pkg=cfg.pkg,
            img_size=cfg.img_size,
            relevant_layers=cfg.layers,
            pooling=cfg.pooling,
            dtype=convert_dtype_by_name(cfg.dtype, "torch"),
            repo_url=model_source,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
            device=get_device(),
        )
        print(f"prepared {model_source}")
        return
    # end if cfg.prepare_only:

    _, rank, size = parallel_setup()
    layer_names = cfg.layers or get_relevant_output_layers(cfg.model_name, cfg.pkg)
    if not layer_names:
        raise ValueError("At least one layer must be selected for extraction.")
    # end if not layer_names:
    if rank == 0:
        print_wise(cfg)
    # end if rank == 0:
    if size == 1:
        device = get_device()
        ann, loader, model_input_key = load_feature_extraction_inputs(
            paths, cfg, device
        )
        print_wise(f"loaded {cfg.model_name} and {len(loader.dataset)} images")
        extract_layer_group(
            paths, rank, layer_names, ann, loader, model_input_key, cfg
        )
        return
    # end if size == 1:

    layer_groups = split_layers(layer_names, size - 1)
    if rank == 0:
        ann, loader, model_input_key = None, None, None
    else:
        n_threads = max(1, (os.cpu_count() or 1) // (size - 1))
        torch.set_num_threads(n_threads)
        device = get_device()
        ann, loader, model_input_key = load_feature_extraction_inputs(
            paths, cfg, device
        )
        print_wise(
            f"loaded {cfg.model_name} and {len(loader.dataset)} images on {device}",
            rank=rank,
        )
    # end if rank == 0:
    master_workers_queue(
        layer_groups,
        paths,
        extract_layer_group,
        ann,
        loader,
        model_input_key,
        cfg,
    )
# EOF
