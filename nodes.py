import importlib.util
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
BIREFNET_HF_REPO = os.environ.get("ZCUT_BIREFNET_HF_REPO", "1038lab/BiRefNet")
SAM3_HF_REPOS = [repo.strip() for repo in os.environ.get("ZCUT_SAM3_HF_REPOS", "facebook/sam3,AB498/sam3").split(",") if repo.strip()]

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


def _numpy_batch_to_tensor(images):
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _mask_batch_to_tensor(masks):
    arr = np.stack(masks, axis=0).astype(np.float32)
    return torch.from_numpy(arr)


def _resize_mask(mask, width, height):
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def _shape_mask(width, height, shape):
    mask = np.ones((height, width), dtype=np.float32)
    if shape == "circle":
        yy, xx = np.ogrid[:height, :width]
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        radius = min(width, height) * 0.5
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


def _rgba_from_rgb_and_alpha(rgb, alpha):
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.dstack((rgb, alpha_u8))


def _resize_to_canvas(arr, output_width, output_height, interpolation, pad_value=0):
    height, width = arr.shape[:2]
    scale = min(output_width / max(width, 1), output_height / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
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


def _resize_image_batch_to_canvas(image, output_width, output_height):
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None,]
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    resized = [
        _resize_to_canvas(img, output_width, output_height, cv2.INTER_AREA if img.shape[0] > output_height else cv2.INTER_LINEAR)
        for img in arr
    ]
    return torch.from_numpy(np.stack(resized, axis=0).astype(np.float32))


def _resize_mask_batch_to_canvas(mask, output_width, output_height):
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[None,]
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    resized = [_resize_to_canvas(single, output_width, output_height, cv2.INTER_LINEAR) for single in arr]
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cropped_image", "crop_mask", "birefnet_mask")
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
    ):
        cropped_images = []
        crop_masks = []
        birefnet_masks = []
        shape_mask = _shape_mask(crop_width, crop_height, shape)
        if feather_edges:
            shape_mask = np.clip(_feather_shape_mask(shape_mask, feather_px), 0.0, 1.0)

        for rgb in _tensor_to_numpy_batch(image):
            subject_mask = _run_birefnet(rgb, birefnet_model, birefnet_threshold, device)
            center = _choose_face_center(rgb, subject_mask, sam3_prompt, sam3_confidence, device)
            cropped_rgb, crop_box = _crop_around_center(rgb, center, crop_width, crop_height)
            cropped_subject = _crop_mask(subject_mask, crop_box, crop_width, crop_height)
            cropped_subject = _clean_subject_mask(cropped_subject)
            final_mask = np.clip(cropped_subject * shape_mask, 0.0, 1.0)
            cropped_images.append(_rgba_from_rgb_and_alpha(cropped_rgb, final_mask))
            crop_masks.append(final_mask)
            birefnet_masks.append(subject_mask)

        return (
            _numpy_batch_to_tensor(cropped_images),
            _mask_batch_to_tensor(crop_masks),
            _mask_batch_to_tensor(birefnet_masks),
        )


class ZcutResizeOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "output_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("resized_image", "resized_mask")
    FUNCTION = "run"
    CATEGORY = "Zcut"

    def run(self, image, output_width, output_height, mask=None):
        resized_image = _resize_image_batch_to_canvas(image, output_width, output_height)
        source_mask = mask if mask is not None else _alpha_mask_from_image(image)
        resized_mask = _resize_mask_batch_to_canvas(source_mask, output_width, output_height)
        return resized_image, resized_mask


NODE_CLASS_MAPPINGS = {
    "ZcutBiRefNetSAM3FaceCrop": ZcutBiRefNetSAM3FaceCrop,
    "ZcutResizeOutput": ZcutResizeOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZcutBiRefNetSAM3FaceCrop": "Zcut BiRefNet SAM3 Face Crop",
    "ZcutResizeOutput": "Zcut Resize Output",
}
