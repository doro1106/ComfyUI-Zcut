# ComfyUI-Zcut

**English** | [中文](#中文)

ComfyUI-Zcut is a ComfyUI custom node plugin for portrait cutout, face-centered avatar cropping, circular/square alpha masks, and final output resizing.

It combines BiRefNet foreground extraction with SAM3-based face/head localization. The crop output is designed to behave like a Photoshop layer mask workflow: first make a clean subject cutout, then apply a feathered circular or square crop mask without stretching or smearing subject pixels into the transparent background.

## Features

- Portrait foreground cutout powered by BiRefNet.
- Face/head-centered crop assisted by SAM3 text-prompt segmentation.
- Square/rectangle and circular transparent avatar output.
- Photoshop-style crop edge feathering: only the crop shape edge is feathered, not the subject edge.
- Final resize node that scales uniformly into a transparent canvas without changing the crop shape, aspect ratio, alpha, or transparent background.
- Automatic model download on first use when model files are missing and the environment has Hugging Face access.

## Installation

Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/doro1106/ComfyUI-Zcut.git
```

Restart ComfyUI after installation.

The plugin attempts to install missing Python dependencies when ComfyUI imports it. You can also install them manually in your ComfyUI Python environment:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Zcut
pip install -r requirements.txt
```

`torch` and `torchvision` should come from your existing ComfyUI environment. Do not install a separate PyTorch build from this plugin unless your ComfyUI setup requires it.

## Models

Model files are **not included** in this GitHub repository.

On first use, the node can download the required files from Hugging Face:

- BiRefNet files from `1038lab/BiRefNet`
- SAM3 files from `facebook/sam3`, with `AB498/sam3` as a fallback source

For offline installation, place the files manually:

```text
ComfyUI/custom_nodes/ComfyUI-Zcut/models/BiRefNet/*.safetensors
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/sam3.pt
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

If you use private or mirrored Hugging Face repositories, set these environment variables before starting ComfyUI:

```bash
ZCUT_BIREFNET_HF_REPO=1038lab/BiRefNet
ZCUT_SAM3_HF_REPOS=facebook/sam3,AB498/sam3
```

## Nodes

### Zcut BiRefNet SAM3 Face Crop

Creates a transparent portrait/avatar crop.

Inputs:

- `image`: ComfyUI image input.
- `birefnet_model`: BiRefNet model variant.
- `crop_width`, `crop_height`: Output crop size in pixels.
- `shape`: `square` keeps the crop rectangle; `circle` applies a circular alpha mask.
- `feather_edges`: Enables square/circle crop-edge feathering.
- `feather_px`: Crop-edge feather radius in pixels. This does not feather the subject edge.
- `device`: Auto, GPU, or CPU.
- `birefnet_threshold`: Foreground mask threshold.
- `sam3_prompt`: Comma-separated prompts for SAM3 face/head localization.
- `sam3_confidence`: SAM3 confidence threshold.

Outputs:

- `cropped_image`: RGBA image with transparent background.
- `crop_mask`: Final alpha mask.
- `birefnet_mask`: Original BiRefNet foreground mask.

### Zcut Resize Output

Resizes the final avatar after cropping.

Inputs:

- `image`: Cropped RGBA image.
- `output_width`, `output_height`: Final canvas size in pixels.
- `mask`: Optional mask to resize together with the image. If omitted, the image alpha channel is used.

Outputs:

- `resized_image`: Image scaled uniformly and centered on a transparent canvas.
- `resized_mask`: Mask scaled with the same geometry.

## Notes

- SAM3 is a segmentation model, not a dedicated face detector. For unusual poses or stylized images, adjust the crop size or prompt.
- `feather_px` affects the square/circle crop mask only. It does not blur or stretch the foreground subject edge.
- Transparent RGB pixels are not filled by dragging subject edge pixels outward.

## Credits

Special thanks to the model authors and maintainers:

- BiRefNet by 1038lab: https://huggingface.co/1038lab/BiRefNet
- SAM3 by Meta AI: https://huggingface.co/facebook/sam3
- SAM3 fallback mirror used by this plugin: https://huggingface.co/AB498/sam3/tree/main

This repository contains only plugin code. Please follow the licenses and usage terms of the referenced models when downloading or using their weights.

## 中文

ComfyUI-Zcut 是一个用于 ComfyUI 的自定义节点插件，用于人物抠图、头像居中裁切、圆形/方形透明蒙版裁切，以及最终输出尺寸调整。

插件结合 BiRefNet 前景抠图和 SAM3 人脸/头部定位。输出逻辑接近 Photoshop 图层蒙版流程：先得到干净的人物抠图，再套用带羽化边缘的圆形或方形裁切蒙版，不会把人物边缘像素硬拉扯到透明背景里。

## 功能特点

- 使用 BiRefNet 进行人物前景抠图。
- 使用 SAM3 文本提示分割辅助定位人脸/头部区域。
- 支持方形/矩形和圆形透明头像输出。
- 类 Photoshop 的裁切边缘羽化：只羽化圆形/方形裁切边缘，不羽化人物边缘。
- 提供最终尺寸调整节点，等比缩放到透明画布中，不改变形状、比例、alpha 或透明背景。
- 如果模型文件缺失，并且环境可以访问 Hugging Face，首次使用时会自动下载模型。

## 安装

将仓库克隆到 ComfyUI 的自定义节点目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/doro1106/ComfyUI-Zcut.git
```

安装后重启 ComfyUI。

插件在被 ComfyUI 导入时会尝试自动安装缺失依赖。也可以在 ComfyUI 的 Python 环境中手动安装：

```bash
cd ComfyUI/custom_nodes/ComfyUI-Zcut
pip install -r requirements.txt
```

`torch` 和 `torchvision` 应使用 ComfyUI 现有环境中的版本。除非你的 ComfyUI 环境本身需要，否则不要从本插件单独安装另一套 PyTorch。

## 模型文件

GitHub 仓库中**不包含模型文件**。

首次使用时，节点可以从 Hugging Face 自动下载所需文件：

- BiRefNet 文件来自 `1038lab/BiRefNet`
- SAM3 文件优先来自 `facebook/sam3`，并使用 `AB498/sam3` 作为备用来源

如果需要离线安装，请手动放置模型文件：

```text
ComfyUI/custom_nodes/ComfyUI-Zcut/models/BiRefNet/*.safetensors
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/sam3.pt
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

如果使用私有或镜像 Hugging Face 仓库，请在启动 ComfyUI 前设置环境变量：

```bash
ZCUT_BIREFNET_HF_REPO=1038lab/BiRefNet
ZCUT_SAM3_HF_REPOS=facebook/sam3,AB498/sam3
```

## 节点说明

### Zcut BiRefNet SAM3 Face Crop

生成透明背景的人像/头像裁切图。

输入：

- `image`：ComfyUI 图像输入。
- `birefnet_model`：BiRefNet 模型版本。
- `crop_width`、`crop_height`：输出裁切尺寸，单位为像素。
- `shape`：`square` 保留矩形裁切；`circle` 应用圆形 alpha 蒙版。
- `feather_edges`：启用方形/圆形裁切边缘羽化。
- `feather_px`：裁切边缘羽化半径，单位为像素。不会羽化人物边缘。
- `device`：Auto、GPU 或 CPU。
- `birefnet_threshold`：前景 mask 阈值。
- `sam3_prompt`：用于 SAM3 定位脸部/头部的逗号分隔提示词。
- `sam3_confidence`：SAM3 置信度阈值。

输出：

- `cropped_image`：带透明背景的 RGBA 图像。
- `crop_mask`：最终 alpha mask。
- `birefnet_mask`：原始 BiRefNet 前景 mask。

### Zcut Resize Output

在裁切完成后调整最终头像输出尺寸。

输入：

- `image`：裁切后的 RGBA 图像。
- `output_width`、`output_height`：最终画布尺寸，单位为像素。
- `mask`：可选，同步调整的 mask。如果不连接，会使用图像 alpha 通道。

输出：

- `resized_image`：等比缩放并居中放入透明画布的图像。
- `resized_mask`：使用相同几何变换后的 mask。

## 注意事项

- SAM3 是分割模型，不是专用人脸检测器。遇到特殊姿态或风格化图片时，可以调整裁切尺寸或提示词。
- `feather_px` 只影响方形/圆形裁切蒙版，不会模糊或拉伸人物边缘。
- 透明区域的 RGB 不会通过拉扯人物边缘像素来填充。

## 致谢

特别感谢以下模型的作者和维护者：

- BiRefNet by 1038lab: https://huggingface.co/1038lab/BiRefNet
- SAM3 by Meta AI: https://huggingface.co/facebook/sam3
- 本插件使用的 SAM3 备用镜像: https://huggingface.co/AB498/sam3/tree/main

本仓库只包含插件代码。下载或使用模型权重时，请遵守对应模型的许可证和使用条款。
