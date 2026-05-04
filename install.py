import importlib.util
import subprocess
import sys

REQUIREMENTS = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("safetensors", "safetensors"),
    ("timm", "timm"),
    ("transformers", "transformers>=4.39.0"),
    ("huggingface_hub", "huggingface-hub>=0.19.0"),
    ("hf_xet", "hf-xet"),
    ("iopath", "iopath>=0.1.9"),
    ("ftfy", "ftfy"),
    ("regex", "regex"),
    ("typing_extensions", "typing_extensions"),
    ("scipy", "scipy"),
]


def ensure_requirements():
    missing = [package for module, package in REQUIREMENTS if importlib.util.find_spec(module) is None]
    if not missing:
        return
    print(f"[Zcut] Installing missing dependencies: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
