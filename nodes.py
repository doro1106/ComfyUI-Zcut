import importlib.util
import logging
import math
import os
import shutil
import sys
import types
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

PLUGIN_DIR = Path(__file__).resolve().parent
BIREFNET_DIR = PLUGIN_DIR / "models" / "BiRefNet"
SAM3_DIR = PLUGIN_DIR / "models" / "sam3"
SAM3_CHECKPOINT = SAM3_DIR / "sam3.pt"
SAM3_BPE_PATH = SAM3_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"
SAM3_SOURCE_SENTINELS = (
    SAM3_DIR / "__init__.py",
    SAM3_DIR / "model_builder.py",
    SAM3_DIR / "model" / "sam3_image_processor.py",
)
BIREFNET_HF_REPO = os.environ.get("ZCUT_BIREFNET_HF_REPO", "1038lab/BiRefNet")
SAM3_HF_REPOS = [repo.strip() for repo in os.environ.get("ZCUT_SAM3_HF_REPOS", "facebook/sam3,AB498/sam3").split(",") if repo.strip()]
UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
RESIZE_OUTPUT_RESAMPLE_METHODS = ["auto", *UPSCALE_METHODS]
LARGE_MODEL_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".safetensors"}

BIREFNET_MODEL_CANDIDATES = {
    "BiRefNet-portrait": ("birefnet.py", "BiRefNet-portrait.safetensors", 1024),
    "BiRefNet-matting": ("birefnet.py", "BiRefNet-matting.safetensors", 1024),
    "BiRefNet-general": ("birefnet.py", "BiRefNet-general.safetensors", 1024),
    "BiRefNet-HR-matting": ("birefnet.py", "BiRefNet-HR-matting.safetensors", 2048),
    "BiRefNet_512x512": ("birefnet.py", "BiRefNet_512x512.safetensors", 512),
    "BiRefNet_dynamic": ("birefnet.py", "BiRefNet_dynamic.safetensors", 1024),
}


def _available_birefnet_models():
    return dict(BIREFNET_MODEL_CANDIDATES)


BIREFNET_MODELS = _available_birefnet_models()
DEFAULT_BIREFNET_MODEL = "BiRefNet-general" if "BiRefNet-general" in BIREFNET_MODELS else next(iter(BIREFNET_MODELS), "")


def _device_from_choice(choice):
    if choice == "CPU":
        return torch.device("cpu")
    if choice == "GPU":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was selected, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_numpy_batch(image):
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None,]
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _tensor_to_rgb_alpha_batch(image):
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None,]
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

    if arr.shape[-1] >= 4:
        alpha = arr[..., 3]
    else:
        alpha = np.ones(arr.shape[:3], dtype=np.float32)

    rgb = arr[..., :3]
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    elif rgb.shape[-1] < 3:
        pad = np.zeros((*rgb.shape[:-1], 3 - rgb.shape[-1]), dtype=rgb.dtype)
        rgb = np.concatenate((rgb, pad), axis=-1)

    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8), np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _tensor_has_alpha_channel(image):
    arr = image.detach().cpu()
    return arr.ndim >= 3 and arr.shape[-1] >= 4


def _image_to_rgb_alpha_for_upscale(image):
    rgb_batch, image_alpha = _tensor_to_rgb_alpha_batch(image)
    return rgb_batch, image_alpha, _tensor_has_alpha_channel(image)


def _numpy_batch_to_tensor(images):
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _mask_batch_to_tensor(masks):
    arr = np.stack(masks, axis=0).astype(np.float32)
    return torch.from_numpy(arr)


def _mask_to_numpy_batch(mask):
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[None,]
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _resize_mask(mask, width, height):
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def _cv2_interpolation(method):
    if method == "nearest-exact":
        return getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    if method == "bilinear":
        return cv2.INTER_LINEAR
    if method == "area":
        return cv2.INTER_AREA
    if method == "bicubic":
        return cv2.INTER_CUBIC
    if method == "lanczos":
        return cv2.INTER_LANCZOS4
    return cv2.INTER_LINEAR


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _available_upscale_models():
    try:
        import folder_paths

        models = folder_paths.get_filename_list("upscale_models")
        return models or ["none"]
    except Exception:
        return ["none"]


@lru_cache(maxsize=4)
def _load_upscale_model(model_name):
    if model_name == "none":
        raise ValueError("No upscale model is selected or available.")

    import comfy.utils
    import folder_paths
    from spandrel import ImageModelDescriptor, ModelLoader

    try:
        from spandrel import MAIN_REGISTRY
        from spandrel_extra_arches import EXTRA_REGISTRY

        MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    except Exception:
        logging.debug("[Zcut] spandrel_extra_arches is not available.", exc_info=True)

    model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
    state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
        state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"module.": ""})

    model = ModelLoader().load_from_state_dict(state_dict).eval()
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    return model


def _apply_upscale_model(upscale_model, image):
    import comfy.model_management as model_management
    import comfy.utils

    device = model_management.get_torch_device()
    memory_required = model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)
    tile = 512
    overlap = 32
    output_device = model_management.intermediate_device()

    oom = True
    try:
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                )
                progress = comfy.utils.ProgressBar(steps)
                scaled = comfy.utils.tiled_scale(
                    in_img,
                    lambda tile_image: upscale_model(tile_image.float()),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=progress,
                    output_device=output_device,
                )
                oom = False
            except Exception as exc:
                model_management.raise_non_oom(exc)
                tile //= 2
                if tile < 128:
                    raise exc
    finally:
        upscale_model.to("cpu")

    return torch.clamp(scaled.movedim(-3, -1), min=0.0, max=1.0).detach().cpu()


def _path_is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _download_hf_file(repo_id, filename, target_path):
    from huggingface_hub import hf_hub_download

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Zcut] Downloading {repo_id}/{filename} -> {target_path}")
    cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    if cached_path.resolve() != target_path.resolve():
        shutil.copy2(cached_path, target_path)
    return target_path


def _copy_hf_snapshot_without_weights(snapshot_path, target_dir):
    snapshot_path = Path(snapshot_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in snapshot_path.rglob("*"):
        if not source_path.is_file():
            continue
        if source_path.suffix.lower() in LARGE_MODEL_SUFFIXES:
            continue
        relative_path = source_path.relative_to(snapshot_path)
        target_path = target_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _sam3_source_is_available():
    return all(path.exists() and path.stat().st_size > 0 for path in SAM3_SOURCE_SENTINELS)


def _ensure_sam3_source_files():
    if _sam3_source_is_available():
        return

    from huggingface_hub import snapshot_download

    errors = []
    allow_patterns = [
        "*.py",
        "assets/*",
        "model/**/*.py",
        "perflib/**/*.py",
        "sam/**/*.py",
        "train/**/*.py",
    ]
    ignore_patterns = ["*.bin", "*.ckpt", "*.onnx", "*.pt", "*.safetensors", "__pycache__/*"]
    for repo_id in SAM3_HF_REPOS:
        try:
            print(f"[Zcut] Downloading SAM3 runtime source from {repo_id} -> {SAM3_DIR}")
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            _copy_hf_snapshot_without_weights(snapshot_path, SAM3_DIR)
            if _sam3_source_is_available():
                return
            missing = ", ".join(str(path.relative_to(SAM3_DIR)) for path in SAM3_SOURCE_SENTINELS if not path.exists())
            errors.append(f"{repo_id}: snapshot did not contain required files: {missing}")
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "[Zcut] Failed to prepare SAM3 runtime source files. Check network access, Hugging Face access permissions, "
        f"or set ZCUT_SAM3_HF_REPOS to a mirror that includes the SAM3 Python source.\n{joined}"
    )


def _ensure_hf_file(repo_id, filename, target_path):
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    try:
        return _download_hf_file(repo_id, filename, target_path)
    except Exception as exc:
        raise RuntimeError(
            f"[Zcut] Failed to download {repo_id}/{filename}. Check network access, Hugging Face access permissions, "
            "or set the related ZCUT_*_HF_REPO environment variable."
        ) from exc


def _ensure_birefnet_files(model_name):
    if model_name not in BIREFNET_MODELS:
        available = ", ".join(BIREFNET_MODELS) or "none"
        raise ValueError(f"Unknown BiRefNet model: {model_name}. Available models: {available}")

    model_file, weights_file, _ = BIREFNET_MODELS[model_name]
    for filename in ("BiRefNet_config.py", model_file, "config.json", weights_file):
        _ensure_hf_file(BIREFNET_HF_REPO, filename, BIREFNET_DIR / filename)


def _ensure_sam3_files():
    _ensure_sam3_source_files()

    missing = []
    if not SAM3_CHECKPOINT.exists() or SAM3_CHECKPOINT.stat().st_size == 0:
        missing.append(("sam3.pt", SAM3_CHECKPOINT))
    if not SAM3_BPE_PATH.exists() or SAM3_BPE_PATH.stat().st_size == 0:
        missing.append(("bpe_simple_vocab_16e6.txt.gz", SAM3_BPE_PATH))
    if not missing:
        return

    errors = []
    for filename, target_path in missing:
        downloaded = False
        for repo_id in SAM3_HF_REPOS:
            try:
                _download_hf_file(repo_id, filename, target_path)
                downloaded = True
                break
            except Exception as exc:
                errors.append(f"{repo_id}/{filename}: {exc}")
        if not downloaded:
            joined = "\n".join(errors)
            raise RuntimeError(
                "[Zcut] Failed to download SAM3 files. The official facebook/sam3 repo may require Hugging Face login "
                f"and access approval. Tried: {', '.join(SAM3_HF_REPOS)}\n{joined}"
            )


@lru_cache(maxsize=3)
def _load_birefnet(model_name, device_type):
    _ensure_birefnet_files(model_name)
    if model_name not in BIREFNET_MODELS:
        available = ", ".join(BIREFNET_MODELS) or "none"
        raise ValueError(f"Unknown or missing BiRefNet model: {model_name}. Available bundled models: {available}")
    model_file, weights_file, process_res = BIREFNET_MODELS[model_name]
    config_path = BIREFNET_DIR / "BiRefNet_config.py"
    model_path = BIREFNET_DIR / model_file
    weights_path = BIREFNET_DIR / weights_file

    for path in (config_path, model_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing BiRefNet file: {path}")

    package_name = "ZcutBiRefNetLocal"
    package = types.ModuleType(package_name)
    package.__path__ = [str(BIREFNET_DIR)]
    sys.modules[package_name] = package
    config_module = _load_module(f"{package_name}.BiRefNet_config", config_path)
    model_module = _load_module(f"{package_name}.{model_path.stem}", model_path)
    model = model_module.BiRefNet(config_module.BiRefNetConfig())
    model.load_state_dict(load_file(str(weights_path)))
    model.eval()

    device = torch.device(device_type)
    model = model.to(device)
    if device.type == "cuda":
        model = model.half()
    return model, process_res


def _run_birefnet(rgb, model_name, threshold, device_choice):
    height, width = rgb.shape[:2]
    device = _device_from_choice(device_choice)
    model, process_res = _load_birefnet(model_name, device.type)

    pil = Image.fromarray(rgb)
    transform_image = transforms.Compose(
        [
            transforms.Resize((process_res, process_res), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform_image(pil).unsqueeze(0).to(device)
    if device.type == "cuda":
        tensor = tensor.half()

    with torch.inference_mode():
        out = model(tensor)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        pred = torch.sigmoid(pred.float())
        pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
        mask = pred[0, 0].detach().cpu().numpy()

    mask = np.clip(mask, 0.0, 1.0)
    if threshold > 0:
        mask = (mask >= threshold).astype(np.float32)
    return mask


@lru_cache(maxsize=2)
def _load_sam3(device_type):
    _ensure_sam3_files()

    sam3_parent = str(SAM3_DIR.parent)
    if sam3_parent in sys.path:
        sys.path.remove(sam3_parent)
    sys.path.insert(0, sam3_parent)

    has_foreign_sam3 = False
    for name, module in list(sys.modules.items()):
        if name != "sam3" and not name.startswith("sam3."):
            continue
        module_file_attr = getattr(module, "__file__", None)
        if module_file_attr and not _path_is_relative_to(Path(module_file_attr).resolve(), SAM3_DIR):
            has_foreign_sam3 = True
            break
    if has_foreign_sam3:
        for name in list(sys.modules):
            if name == "sam3" or name.startswith("sam3."):
                del sys.modules[name]

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=SAM3_BPE_PATH,
        device=device_type,
        eval_mode=True,
        checkpoint_path=SAM3_CHECKPOINT,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    return Sam3Processor(model, device=device_type)


def _sam3_masks(rgb, prompt, confidence_threshold, device_choice):
    device = _device_from_choice(device_choice)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    processor = _load_sam3(device_type)
    pil = Image.fromarray(rgb)
    prompts = [p.strip() for p in prompt.replace("，", ",").split(",") if p.strip()]
    if not prompts:
        prompts = ["face", "head", "anime face", "game character face"]

    autocast_enabled = device.type == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast_enabled else nullcontext()
    masks = []
    with ctx:
        state = processor.set_image(pil)
        for text in prompts:
            processor.reset_all_prompts(state)
            processor.set_confidence_threshold(float(confidence_threshold), state)
            state = processor.set_text_prompt(text, state)
            found = state.get("masks")
            logits = state.get("masks_logits")
            if found is None or found.numel() == 0:
                continue
            found = found.float()
            if found.ndim == 4:
                found = found.squeeze(1)
            scores = torch.ones((found.shape[0],), device=found.device)
            if logits is not None:
                logits = logits.float()
                if logits.ndim == 4:
                    logits = logits.squeeze(1)
                scores = logits.mean(dim=(-2, -1))
            for idx in torch.argsort(scores, descending=True)[:12]:
                masks.append(found[idx].clamp(0, 1).detach().cpu().numpy())
    return masks


def _foreground_box(mask):
    ys, xs = np.where(mask > 0.2)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape
        return 0, 0, width, height
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _skin_score(rgb, segmentation):
    if segmentation.sum() == 0:
        return 0.0
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin = (cr > 130) & (cr < 180) & (cb > 75) & (cb < 145) & (y > 40)
    return float((skin & segmentation).sum() / max(segmentation.sum(), 1))


def _choose_face_center(rgb, subject_mask, sam_prompt, confidence_threshold, device_choice):
    x0, y0, x1, y1 = _foreground_box(subject_mask)
    subject_w = max(x1 - x0, 1)
    subject_h = max(y1 - y0, 1)
    subject_area = max(float((subject_mask > 0.2).sum()), 1.0)

    best_score = -1.0
    best_center = None
    for mask in _sam3_masks(rgb, sam_prompt, confidence_threshold, device_choice):
        seg = mask > 0.5
        area = float(seg.sum())
        if area < 64:
            continue
        overlap = float((seg & (subject_mask > 0.2)).sum())
        if overlap / max(area, 1.0) < 0.35:
            continue

        ys, xs = np.where(seg)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        bw, bh = bx1 - bx0, by1 - by0
        cx, cy = (bx0 + bx1) * 0.5, (by0 + by1) * 0.5
        rel_x = abs(cx - (x0 + x1) * 0.5) / subject_w
        rel_y = (cy - y0) / subject_h
        area_ratio = area / subject_area
        aspect = bw / max(bh, 1)

        upper_score = 1.0 - min(abs(rel_y - 0.22) / 0.35, 1.0)
        center_score = 1.0 - min(rel_x / 0.55, 1.0)
        size_score = 1.0 - min(abs(np.log(max(area_ratio, 1e-4) / 0.08)) / 2.2, 1.0)
        aspect_score = 1.0 - min(abs(aspect - 0.85) / 1.2, 1.0)
        score = upper_score * 2.4 + center_score * 1.4 + size_score * 1.5 + aspect_score * 0.8 + _skin_score(rgb, seg) * 0.5

        if score > best_score:
            best_score = score
            best_center = (cx, cy)

    if best_center is not None:
        return best_center
    return ((x0 + x1) * 0.5, y0 + subject_h * 0.24)


def _crop_around_center(rgb, center, crop_width, crop_height):
    height, width = rgb.shape[:2]
    cx, cy = center
    left = int(round(cx - crop_width * 0.5))
    top = int(round(cy - crop_height * 0.5))
    right = left + crop_width
    bottom = top + crop_height
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    padded = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded[top + pad_top : bottom + pad_top, left + pad_left : right + pad_left], (left, top, right, bottom)


def _crop_mask(mask, box, crop_width, crop_height):
    height, width = mask.shape
    left, top, right, bottom = box
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    padded = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    cropped = padded[top + pad_top : bottom + pad_top, left + pad_left : right + pad_left]
    return _resize_mask(cropped, crop_width, crop_height)


def _shape_mask(width, height, shape, feather_px=0):
    mask = np.ones((height, width), dtype=np.float32)
    feather_margin = max(0, int(feather_px) * 2)
    if shape == "square" and feather_margin > 0:
        left = min(feather_margin, max(width // 2, 0))
        right = max(width - feather_margin, left)
        top = min(feather_margin, max(height // 2, 0))
        bottom = max(height - feather_margin, top)
        mask = np.zeros((height, width), dtype=np.float32)
        mask[top:bottom, left:right] = 1.0
    if shape == "circle":
        yy, xx = np.ogrid[:height, :width]
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        radius = max(0.0, min(width, height) * 0.5 - feather_margin)
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2).astype(np.float32)
    return mask


def _feather(mask, feather_px):
    feather_px = int(feather_px)
    if feather_px <= 0:
        return mask
    k = max(3, feather_px * 2 + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(mask, (k, k), sigmaX=max(feather_px / 2.0, 0.1))


def _feather_shape_mask(mask, feather_px):
    feather_px = int(feather_px)
    if feather_px <= 0:
        return mask
    pad = max(1, feather_px * 2)
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    feathered = _feather(padded, feather_px)
    return feathered[pad : pad + mask.shape[0], pad : pad + mask.shape[1]]


def _clean_subject_mask(mask):
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return (mask >= 0.5).astype(np.float32)


def _nearest_opaque_rgb_fill(rgb, alpha, threshold=0.01):
    rgb = rgb.astype(np.uint8, copy=True)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    opaque = alpha > threshold
    if opaque.all() or not opaque.any():
        rgb[~opaque] = 0
        return rgb

    _, labels = cv2.distanceTransformWithLabels(
        (~opaque).astype(np.uint8),
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    coords = np.column_stack(np.where(opaque))
    nearest = coords[labels - 1]
    filled = rgb.copy()
    transparent = ~opaque
    filled[transparent] = rgb[nearest[transparent, 0], nearest[transparent, 1]]
    return filled


def _replace_edge_rgb_from_interior(rgb, alpha, edge_px=2, threshold=0.5):
    edge_px = int(edge_px)
    if edge_px <= 0:
        return rgb.astype(np.uint8, copy=True)

    rgb = rgb.astype(np.uint8, copy=True)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    opaque = alpha > threshold
    if not opaque.any():
        return rgb

    distance_inside = cv2.distanceTransform(opaque.astype(np.uint8), cv2.DIST_L2, 3)
    edge = opaque & (distance_inside <= edge_px)
    interior = opaque & (distance_inside > edge_px)
    if not edge.any() or not interior.any():
        return rgb

    _, labels = cv2.distanceTransformWithLabels(
        (~interior).astype(np.uint8),
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    coords = np.column_stack(np.where(interior))
    nearest = coords[labels - 1]
    cleaned = rgb.copy()
    cleaned[edge] = rgb[nearest[edge, 0], nearest[edge, 1]]
    return cleaned


def _rgba_from_rgb_and_alpha(rgb, alpha, defringe_px=0, defringe_alpha=None):
    defringe_alpha = alpha if defringe_alpha is None else defringe_alpha
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    clean_rgb = _nearest_opaque_rgb_fill(rgb, alpha)
    clean_rgb = _replace_edge_rgb_from_interior(clean_rgb, defringe_alpha, defringe_px)
    clean_rgb[alpha <= 0.0] = 0
    return np.dstack((clean_rgb, alpha_u8))


def _resize_array_exact(arr, width, height, method):
    return cv2.resize(arr, (int(width), int(height)), interpolation=_cv2_interpolation(method))


def _resize_rgba_exact(rgba, output_width, output_height, method):
    alpha = np.clip(rgba[..., 3], 0.0, 1.0).astype(np.float32)
    rgb = np.clip(rgba[..., :3], 0.0, 1.0).astype(np.float32)
    premultiplied = rgb * alpha[..., None]

    resized_rgb = _resize_array_exact(premultiplied, output_width, output_height, method)
    resized_alpha = _resize_array_exact(alpha, output_width, output_height, method)
    safe_alpha = np.maximum(resized_alpha, 1e-6)
    straight_rgb = resized_rgb / safe_alpha[..., None]
    straight_rgb[resized_alpha <= 1e-6] = 0.0

    return np.dstack((np.clip(straight_rgb, 0.0, 1.0), np.clip(resized_alpha, 0.0, 1.0))).astype(np.float32)


def _resize_rgb_exact(rgb, output_width, output_height, method):
    return np.clip(_resize_array_exact(rgb, output_width, output_height, method), 0.0, 1.0).astype(np.float32)


def _canvas_resize_interpolation(width, height, resized_width, resized_height, resample_method):
    if resample_method == "auto":
        if resized_width < width or resized_height < height:
            return cv2.INTER_AREA
        return cv2.INTER_LANCZOS4
    return _cv2_interpolation(resample_method)


def _resize_to_canvas(arr, output_width, output_height, resample_method, pad_value=0):
    height, width = arr.shape[:2]
    scale = min(output_width / max(width, 1), output_height / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = _canvas_resize_interpolation(width, height, resized_width, resized_height, resample_method)
    resized = cv2.resize(arr, (resized_width, resized_height), interpolation=interpolation)

    if arr.ndim == 2:
        canvas = np.full((output_height, output_width), pad_value, dtype=arr.dtype)
    else:
        channels = arr.shape[2]
        canvas = np.full((output_height, output_width, channels), pad_value, dtype=arr.dtype)

    left = (output_width - resized_width) // 2
    top = (output_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _resize_rgba_to_canvas(img, output_width, output_height, resample_method):
    alpha = np.clip(img[..., 3], 0.0, 1.0)
    rgb = np.clip(img[..., :3], 0.0, 1.0)
    premultiplied = rgb * alpha[..., None]

    resized_rgb = _resize_to_canvas(premultiplied, output_width, output_height, resample_method)
    resized_alpha = _resize_to_canvas(alpha, output_width, output_height, resample_method)
    safe_alpha = np.maximum(resized_alpha, 1e-6)
    straight_rgb = resized_rgb / safe_alpha[..., None]
    straight_rgb[resized_alpha <= 1e-6] = 0.0

    return np.dstack((np.clip(straight_rgb, 0.0, 1.0), np.clip(resized_alpha, 0.0, 1.0)))


def _resize_image_batch_to_canvas(image, output_width, output_height, resample_method):
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None,]
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    resized = [
        _resize_rgba_to_canvas(img, output_width, output_height, resample_method)
        if img.shape[-1] >= 4
        else _resize_to_canvas(img, output_width, output_height, resample_method)
        for img in arr
    ]
    return torch.from_numpy(np.stack(resized, axis=0).astype(np.float32))


def _resize_mask_batch_to_canvas(mask, output_width, output_height, resample_method):
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[None,]
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    resized = [_resize_to_canvas(single, output_width, output_height, resample_method) for single in arr]
    return torch.from_numpy(np.stack(resized, axis=0).astype(np.float32))


def _alpha_mask_from_image(image):
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None,]
    if arr.shape[-1] >= 4:
        alpha = arr[..., 3]
    else:
        alpha = np.ones(arr.shape[:3], dtype=np.float32)
    return torch.from_numpy(np.clip(alpha, 0.0, 1.0).astype(np.float32))


class ZcutBiRefNetSAM3FaceCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "birefnet_model": (list(BIREFNET_MODELS.keys()), {"default": DEFAULT_BIREFNET_MODEL}),
                "crop_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "crop_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "shape": (["square", "circle"], {"default": "square"}),
                "feather_edges": ("BOOLEAN", {"default": True}),
                "feather_px": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "device": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
                "birefnet_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam3_prompt": ("STRING", {"default": "face, head, anime face, game character face"}),
                "sam3_confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01}),
                "enable_cutout": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "source_transparency_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("cropped_image", "crop_mask", "birefnet_mask", "shape_mask")
    FUNCTION = "run"
    CATEGORY = "Zcut"

    def run(
        self,
        image,
        birefnet_model,
        crop_width,
        crop_height,
        shape,
        feather_edges,
        feather_px,
        device,
        birefnet_threshold,
        sam3_prompt,
        sam3_confidence,
        enable_cutout,
        source_transparency_mask=None,
    ):
        cropped_images = []
        crop_masks = []
        birefnet_masks = []
        shape_masks = []
        shape_feather_px = feather_px if feather_edges else 0
        shape_mask = _shape_mask(crop_width, crop_height, shape, shape_feather_px)
        if feather_edges:
            shape_mask = np.clip(_feather_shape_mask(shape_mask, feather_px), 0.0, 1.0)

        rgb_batch, alpha_batch = _tensor_to_rgb_alpha_batch(image)
        transparency_batch = _mask_to_numpy_batch(source_transparency_mask) if source_transparency_mask is not None else None

        for idx, (rgb, image_alpha) in enumerate(zip(rgb_batch, alpha_batch)):
            source_alpha = image_alpha
            if transparency_batch is not None:
                transparency_mask = transparency_batch[min(idx, len(transparency_batch) - 1)]
                if transparency_mask.shape != image_alpha.shape:
                    transparency_mask = _resize_mask(transparency_mask, rgb.shape[1], rgb.shape[0])
                source_alpha = 1.0 - np.clip(transparency_mask, 0.0, 1.0)

            if enable_cutout:
                birefnet_mask = _run_birefnet(rgb, birefnet_model, birefnet_threshold, device)
                subject_mask = np.clip(birefnet_mask * source_alpha, 0.0, 1.0).astype(np.float32)
            else:
                subject_mask = np.clip(source_alpha, 0.0, 1.0).astype(np.float32)
                birefnet_mask = subject_mask

            center = _choose_face_center(rgb, subject_mask, sam3_prompt, sam3_confidence, device)
            cropped_rgb, crop_box = _crop_around_center(rgb, center, crop_width, crop_height)
            cropped_source_alpha = _crop_mask(source_alpha, crop_box, crop_width, crop_height)
            cropped_base_mask = _crop_mask(subject_mask, crop_box, crop_width, crop_height)
            if enable_cutout:
                cropped_base_mask = _clean_subject_mask(cropped_base_mask)
            final_mask = np.clip(cropped_base_mask * shape_mask, 0.0, 1.0)
            defringe_alpha = cropped_base_mask if enable_cutout else cropped_source_alpha
            defringe_enabled = enable_cutout or np.any((cropped_source_alpha > 0.0) & (cropped_source_alpha < 1.0))
            cropped_images.append(
                _rgba_from_rgb_and_alpha(
                    cropped_rgb,
                    final_mask,
                    defringe_px=2 if defringe_enabled else 0,
                    defringe_alpha=defringe_alpha,
                )
            )
            crop_masks.append(final_mask)
            birefnet_masks.append(birefnet_mask)
            shape_masks.append(shape_mask)

        return (
            _numpy_batch_to_tensor(cropped_images),
            _mask_batch_to_tensor(crop_masks),
            _mask_batch_to_tensor(birefnet_masks),
            _mask_batch_to_tensor(shape_masks),
        )


def _round_dimension_to_multiple(value, multiple):
    value = max(1, int(round(value)))
    multiple = int(multiple)
    if multiple <= 1:
        return value
    return max(1, int(round(value / multiple) * multiple))


def _target_dimensions(width, height, mode, rescale_factor, resize_mode, resize_size, round_to_multiple):
    width = max(1, int(width))
    height = max(1, int(height))

    if mode == "rescale":
        target_width = width * float(rescale_factor)
        target_height = height * float(rescale_factor)
    else:
        resize_size = max(1, int(resize_size))
        if resize_mode == "width":
            target_width = resize_size
            target_height = height * (resize_size / width)
        elif resize_mode == "height":
            target_height = resize_size
            target_width = width * (resize_size / height)
        else:
            longest = max(width, height)
            scale = resize_size / longest
            target_width = width * scale
            target_height = height * scale

    return (
        _round_dimension_to_multiple(target_width, round_to_multiple),
        _round_dimension_to_multiple(target_height, round_to_multiple),
    )


def _clean_rgb_for_alpha(rgb_batch, alpha_batch):
    cleaned = []
    for rgb, alpha in zip(rgb_batch, alpha_batch):
        cleaned.append(_nearest_opaque_rgb_fill(rgb, alpha).astype(np.float32) / 255.0)
    return np.stack(cleaned, axis=0).astype(np.float32)


def _premultiply_rgb_by_alpha(rgb_batch, alpha_batch):
    return np.clip(rgb_batch, 0.0, 1.0).astype(np.float32) * np.clip(alpha_batch, 0.0, 1.0).astype(np.float32)[..., None]


def _unpremultiply_rgb_by_alpha(rgb_batch, alpha_batch, epsilon=1e-6):
    alpha = np.clip(alpha_batch, 0.0, 1.0).astype(np.float32)
    rgb = np.clip(rgb_batch, 0.0, 1.0).astype(np.float32)
    out = rgb / np.maximum(alpha, epsilon)[..., None]
    out[alpha <= epsilon] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _clear_rgb_where_transparent(rgb_batch, alpha_batch, epsilon=1e-6):
    rgb = np.clip(rgb_batch, 0.0, 1.0).astype(np.float32).copy()
    alpha = np.clip(alpha_batch, 0.0, 1.0).astype(np.float32)
    rgb[alpha <= epsilon] = 0.0
    return rgb


def _resize_rgba_batch_exact(rgb_batch, alpha_batch, output_width, output_height, method):
    images = []
    masks = []
    for rgb, alpha in zip(rgb_batch, alpha_batch):
        rgba = np.dstack((np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)))
        resized = _resize_rgba_exact(rgba, output_width, output_height, method)
        resized_rgb = resized[..., :3]
        resized_rgb[resized[..., 3] <= 0.0] = 0.0
        resized[..., :3] = resized_rgb
        images.append(resized)
        masks.append(resized[..., 3])
    return (
        torch.from_numpy(np.stack(images, axis=0).astype(np.float32)),
        torch.from_numpy(np.stack(masks, axis=0).astype(np.float32)),
    )


def _color_to_rgb(color):
    if isinstance(color, str):
        color = color.strip()
        if len(color) != 7 or color[0] != "#":
            raise ValueError("background_color must be #RRGGBB or RGB int.")
        try:
            return np.array(
                [
                    int(color[1:3], 16) / 255.0,
                    int(color[3:5], 16) / 255.0,
                    int(color[5:7], 16) / 255.0,
                ],
                dtype=np.float32,
            )
        except ValueError as exc:
            raise ValueError("background_color must be #RRGGBB or RGB int.") from exc

    color = int(color)
    return np.array(
        [
            ((color >> 16) & 0xFF) / 255.0,
            ((color >> 8) & 0xFF) / 255.0,
            (color & 0xFF) / 255.0,
        ],
        dtype=np.float32,
    )


def _image_tensor_to_rgba_float(image):
    rgb_u8, alpha_batch = _tensor_to_rgb_alpha_batch(image)
    rgb = rgb_u8.astype(np.float32) / 255.0
    return np.concatenate((rgb, alpha_batch[..., None]), axis=-1).astype(np.float32)


def _resize_background_rgba(bg, output_width, output_height, fit):
    bg_height, bg_width = bg.shape[:2]
    output_width = max(1, int(output_width))
    output_height = max(1, int(output_height))

    if fit == "stretch":
        return _resize_rgba_exact(bg, output_width, output_height, "lanczos")

    scale = max(output_width / max(1, bg_width), output_height / max(1, bg_height))
    resized_width = max(1, int(round(bg_width * scale)))
    resized_height = max(1, int(round(bg_height * scale)))
    resized = _resize_rgba_exact(bg, resized_width, resized_height, "lanczos")
    left = max(0, (resized_width - output_width) // 2)
    top = max(0, (resized_height - output_height) // 2)
    return resized[top : top + output_height, left : left + output_width]


def _background_image_batch(background_image, batch_size, output_width, output_height, fit):
    if background_image is None:
        return np.zeros((batch_size, output_height, output_width, 3), dtype=np.float32)

    backgrounds_rgba = _image_tensor_to_rgba_float(background_image)
    backgrounds = []
    for idx in range(batch_size):
        bg_idx = min(idx, len(backgrounds_rgba) - 1)
        resized = _resize_background_rgba(backgrounds_rgba[bg_idx], output_width, output_height, fit)
        rgb = _clear_rgb_where_transparent(resized[None, ..., :3], resized[None, ..., 3])[0]
        backgrounds.append(rgb)
    return np.stack(backgrounds, axis=0).astype(np.float32)


def _mask_to_size_batch(mask, batch_size, width, height):
    if mask is None:
        return np.ones((batch_size, height, width), dtype=np.float32)

    mask_batch = _mask_to_numpy_batch(mask)
    fixed = []
    for idx in range(batch_size):
        single = mask_batch[min(idx, len(mask_batch) - 1)]
        if single.shape != (height, width):
            single = _resize_mask(single, width, height)
        fixed.append(np.clip(single, 0.0, 1.0))
    return np.stack(fixed, axis=0).astype(np.float32)


def _composite_subject_over_background_mask(subject_rgb, subject_alpha, background_rgb, background_mask):
    bg_alpha = np.clip(background_mask, 0.0, 1.0).astype(np.float32)
    src_alpha = np.clip(subject_alpha, 0.0, 1.0).astype(np.float32)
    out_alpha = src_alpha + bg_alpha * (1.0 - src_alpha)
    premultiplied = subject_rgb * src_alpha[..., None] + background_rgb * bg_alpha[..., None] * (1.0 - src_alpha[..., None])
    out_rgb = premultiplied / np.maximum(out_alpha, 1e-6)[..., None]
    out_rgb[out_alpha <= 1e-6] = 0.0
    return np.clip(out_rgb, 0.0, 1.0).astype(np.float32), np.clip(out_alpha, 0.0, 1.0).astype(np.float32)


class ZcutModelUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (_available_upscale_models(),),
                "mode": (["rescale", "resize"], {"default": "rescale"}),
                "rescale_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resize_mode": (["longest_side", "width", "height"], {"default": "longest_side"}),
                "resize_size": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "resample_method": (UPSCALE_METHODS, {"default": "lanczos"}),
                "supersample": ("BOOLEAN", {"default": True}),
                "round_to_multiple": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("upscaled_image", "upscaled_mask")
    FUNCTION = "run"
    CATEGORY = "Zcut"

    def run(
        self,
        image,
        upscale_model,
        mode,
        rescale_factor,
        resize_mode,
        resize_size,
        resample_method,
        supersample,
        round_to_multiple,
    ):
        rgb_u8, alpha_batch, has_alpha_input = _image_to_rgb_alpha_for_upscale(image)
        input_height, input_width = rgb_u8.shape[1:3]
        target_width, target_height = _target_dimensions(
            input_width,
            input_height,
            mode,
            rescale_factor,
            resize_mode,
            resize_size,
            round_to_multiple,
        )

        clean_rgb = _clean_rgb_for_alpha(rgb_u8, alpha_batch)
        has_transparency = has_alpha_input or np.any(alpha_batch < 0.999)
        needs_model = target_width > input_width or target_height > input_height

        if needs_model:
            model = _load_upscale_model(upscale_model)
            model_alpha = alpha_batch.astype(np.float32)
            model_rgb_np = _premultiply_rgb_by_alpha(clean_rgb, model_alpha) if has_transparency else clean_rgb.astype(np.float32)
            model_rgb = torch.from_numpy(model_rgb_np.astype(np.float32))

            while True:
                prev_height, prev_width = model_rgb.shape[1:3]
                model_rgb = _apply_upscale_model(model, model_rgb)
                current_height, current_width = model_rgb.shape[1:3]
                model_alpha = np.stack(
                    [
                        _resize_array_exact(alpha, current_width, current_height, resample_method)
                        for alpha in model_alpha
                    ],
                    axis=0,
                ).astype(np.float32)
                if has_transparency:
                    model_rgb_np = _clear_rgb_where_transparent(model_rgb.detach().cpu().numpy(), model_alpha)
                    model_rgb = torch.from_numpy(model_rgb_np.astype(np.float32))
                if (
                    not supersample
                    or (current_width >= target_width and current_height >= target_height)
                    or (current_width <= prev_width and current_height <= prev_height)
                ):
                    break

            output_rgb = model_rgb.detach().cpu().numpy()
            if has_transparency:
                output_rgb = _unpremultiply_rgb_by_alpha(output_rgb, model_alpha)
            result_image, result_mask = _resize_rgba_batch_exact(
                output_rgb,
                model_alpha,
                target_width,
                target_height,
                resample_method,
            )
            return result_image, result_mask

        result_image, result_mask = _resize_rgba_batch_exact(clean_rgb, alpha_batch, target_width, target_height, resample_method)
        return result_image, result_mask


class ZcutResizeOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_output": ("BOOLEAN", {"default": True}),
                "output_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "output_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "resample_method": (RESIZE_OUTPUT_RESAMPLE_METHODS, {"default": "auto"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("resized_image", "resized_mask")
    FUNCTION = "run"
    CATEGORY = "Zcut"

    def run(self, image, resize_output, output_width, output_height, resample_method, mask=None):
        source_mask = mask if mask is not None else _alpha_mask_from_image(image)
        if not resize_output:
            return image, source_mask

        resized_image = _resize_image_batch_to_canvas(image, output_width, output_height, resample_method)
        resized_mask = _resize_mask_batch_to_canvas(source_mask, output_width, output_height, resample_method)
        return resized_image, resized_mask


class ZcutAddBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "background_mode": (["transparent", "image", "color"], {"default": "transparent"}),
                "background_color": ("COLOR", {"default": "#ffffff"}),
                "background_fit": (["cover", "stretch"], {"default": "cover"}),
            },
            "optional": {
                "background_mask": ("MASK",),
                "background_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "Zcut"

    def run(
        self,
        image,
        background_mode,
        background_color,
        background_fit,
        background_mask=None,
        background_image=None,
    ):
        rgba = _image_tensor_to_rgba_float(image)
        alpha = np.clip(rgba[..., 3], 0.0, 1.0).astype(np.float32)
        if not np.any(alpha < 0.999):
            return image, torch.from_numpy(alpha.astype(np.float32))

        rgb = np.clip(rgba[..., :3], 0.0, 1.0).astype(np.float32)
        batch_size, height, width = rgb.shape[:3]
        shape_mask_batch = _mask_to_size_batch(background_mask, batch_size, width, height)

        if background_mode == "transparent":
            rgb = _clear_rgb_where_transparent(rgb, alpha)
            out = np.concatenate((rgb, alpha[..., None]), axis=-1).astype(np.float32)
            return torch.from_numpy(out), torch.from_numpy(alpha.astype(np.float32))

        if background_mode == "image" and background_image is not None:
            background_rgb = _background_image_batch(background_image, batch_size, width, height, background_fit)
        elif background_mode == "image":
            rgb = _clear_rgb_where_transparent(rgb, alpha)
            out = np.concatenate((rgb, alpha[..., None]), axis=-1).astype(np.float32)
            return torch.from_numpy(out), torch.from_numpy(alpha.astype(np.float32))
        else:
            fill_rgb = _color_to_rgb(background_color)
            background_rgb = np.broadcast_to(fill_rgb, (batch_size, height, width, 3)).copy()

        out_rgb, out_alpha = _composite_subject_over_background_mask(rgb, alpha, background_rgb, shape_mask_batch)
        out = np.concatenate((out_rgb, out_alpha[..., None]), axis=-1).astype(np.float32)
        return torch.from_numpy(out), torch.from_numpy(out_alpha)


NODE_CLASS_MAPPINGS = {
    "ZcutBiRefNetSAM3FaceCrop": ZcutBiRefNetSAM3FaceCrop,
    "ZcutAddBackground": ZcutAddBackground,
    "ZcutModelUpscale": ZcutModelUpscale,
    "ZcutResizeOutput": ZcutResizeOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZcutBiRefNetSAM3FaceCrop": "Zcut BiRefNet SAM3 Face Crop",
    "ZcutAddBackground": "Zcut Add Background",
    "ZcutModelUpscale": "Zcut Image Upscale",
    "ZcutResizeOutput": "Zcut Resize Output",
}
