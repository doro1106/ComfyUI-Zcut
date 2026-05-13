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
- Final background node that leaves opaque images unchanged and can keep transparency, add an uploaded image background, or add a picked solid color background.
- Batch image loader node with multi-image upload and folder-path loading modes.
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

The repository also excludes local workflow files and Codex maintenance scratch files: `workflows/` and `codex_maintenance_guide/`.

On first use, the node can download the required files from Hugging Face:

- BiRefNet files from `1038lab/BiRefNet`
- SAM3 runtime source, checkpoint, and vocabulary from `facebook/sam3`, with `AB498/sam3` as a fallback source

The SAM3 runtime source is copied without large model weight files. The checkpoint and vocabulary are downloaded separately into the expected local paths.

For offline installation, place the files manually:

```text
ComfyUI/custom_nodes/ComfyUI-Zcut/models/BiRefNet/*.safetensors
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/__init__.py
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/model_builder.py
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/model/sam3_image_processor.py
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/sam3.pt
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

If you use private or mirrored Hugging Face repositories, set these environment variables before starting ComfyUI:

```bash
ZCUT_BIREFNET_HF_REPO=1038lab/BiRefNet
ZCUT_SAM3_HF_REPOS=facebook/sam3,AB498/sam3
```

## Nodes

### Zcut Batch Load Images

Loads multiple images as a standard ComfyUI image batch that can be connected to other ComfyUI nodes and custom nodes.

Inputs:

- `upload_mode`: `multi_upload` uses the node's multi-image picker button, and `folder_path` reads images from a local folder path.
- `uploaded_images`: Stores the uploaded image references after using the multi-image picker button. You normally do not need to edit this manually.
- `folder_path`: Local folder path used by `folder_path` mode.
- `recursive`: When enabled, `folder_path` mode also scans subfolders.
- `sort_order`: Sorts discovered files by `name` or `modified_time`.
- `size_mode`: Makes images batch-compatible when sizes differ. `pad_to_largest` centers each image on a transparent canvas, `resize_to_first` resizes all images to the first image size, and `skip_mismatched` keeps only images matching the first image size.
- `image_load_limit`: Maximum number of images to load. Use `0` for no limit.

Outputs:

- `images`: Standard RGB `IMAGE` batch.
- `masks`: Standard ComfyUI `MASK` batch. Transparent pixels are represented as masked areas.
- `file_paths`: Newline-separated list of loaded image references or paths.
- `count`: Number of images loaded into the output batch.

Notes:

- Non-image files in `folder_path` mode are ignored.
- Multi-image upload stores selected files in ComfyUI's input directory through the standard `/upload/image` endpoint.
- Animated images use the first frame.

### Zcut BiRefNet SAM3 Face Crop

Creates a transparent portrait/avatar crop.

Inputs:

- `image`: ComfyUI image input.
- `birefnet_model`: BiRefNet model variant.
- `source_transparency_mask`: Optional ComfyUI transparency mask, such as the `mask` output from Load Image. Connect this for transparent PNG inputs when the IMAGE socket does not carry alpha. When cutout is enabled, this alpha is still respected in the final mask.
- `crop_width`, `crop_height`: Output crop size in pixels.
- `shape`: `square` keeps the crop rectangle; `circle` applies a circular alpha mask.
- `feather_edges`: Enables square/circle crop-edge feathering.
- `feather_px`: Crop-edge feather radius in pixels. This does not feather the subject edge. Square and circle crops are inset automatically so the feather can fade to transparent before the canvas edge.
- `device`: Auto, GPU, or CPU.
- `birefnet_threshold`: Foreground mask threshold.
- `sam3_prompt`: Comma-separated prompts for SAM3 face/head localization.
- `sam3_confidence`: SAM3 confidence threshold.
- `enable_cutout`: Enables BiRefNet foreground cutout. When disabled, the node skips BiRefNet, keeps the original image content, preserves the input alpha channel when present, and still applies crop shape masking.

Outputs:

- `cropped_image`: RGBA image with transparent background.
- `crop_mask`: Final alpha mask.
- `birefnet_mask`: Original BiRefNet foreground mask when cutout is enabled; otherwise the base alpha mask used before the crop shape mask.
- `shape_mask`: Square/circle crop shape mask with the same feather settings. Connect this to `Zcut Add Background` `background_mask` when adding a masked background.

### Zcut Resize Output

Resizes the final avatar after cropping.

Inputs:

- `image`: Cropped RGBA image.
- `resize_output`: Enable final canvas resizing. Disable it to pass the image and mask through unchanged.
- `output_width`, `output_height`: Final canvas size in pixels.
- `resample_method`: Resize filter. `auto` uses `area` when scaling down and `lanczos` when scaling up. Manual options are `nearest-exact`, `bilinear`, `area`, `bicubic`, and `lanczos`.
- `mask`: Optional mask to resize together with the image. If omitted, the image alpha channel is used.

Outputs:

- `resized_image`: Image scaled uniformly and centered on a transparent canvas.
- `resized_mask`: Mask scaled with the same geometry.

### Zcut Image Upscale

Upscales or resizes an image with a model from ComfyUI `models/upscale_models`.

Inputs:

- `image`: ComfyUI image input.
- `upscale_model`: Upscale model name from ComfyUI `models/upscale_models`. The workflow stores the relative model name, not an absolute path.
- `mode`: `rescale` scales by a factor; `resize` scales to a target size.
- `rescale_factor`: Scale factor for `rescale` mode.
- `resize_mode`: `longest_side`, `width`, or `height` for `resize` mode.
- `resize_size`: Target longest side, width, or height.
- `resample_method`: Final resampling method.
- `supersample`: If enabled, the model keeps upscaling until the image is at least the target size, then resamples down.
- `round_to_multiple`: Rounds final width and height to this multiple. Use `1` to disable rounding.

Outputs:

- `upscaled_image`: RGBA image. RGB inputs are treated as opaque; RGBA inputs keep alpha.
- `upscaled_mask`: Alpha mask scaled with the same geometry.

Notes:

- Upscale models are used only when the target size is larger than the input. Smaller targets use interpolation resizing.
- Transparent output clears RGB pixels where alpha is fully transparent after model upscale and final resizing.
- Use `Zcut Resize Output` when you want to place the result on a final transparent canvas without changing its aspect ratio.

### Zcut Add Background

Adds a background only when the input image has transparent pixels. If the image is fully opaque, the node returns the original image unchanged.

Inputs:

- `image`: Final image input.
- `background_mode`: `transparent` keeps the transparent background, `image` composites over an uploaded background image, and `color` composites over a picked solid color.
- `background_color`: Solid background color used by `color` mode. This uses ComfyUI's native color picker and eyedropper.
- `background_fit`: How uploaded background images fit the current image size: `cover` automatically scales and crops to fill, and `stretch` stretches to the exact size.
- `background_mask`: Optional mask used to clip the added image/color background. Connect the crop node `shape_mask` output here to reuse the previous square/circle feather settings.
- `background_image`: Optional uploaded background image used when `background_mode` is `image`.

Outputs:

- `image`: Original image when opaque; otherwise RGBA output with the selected background behavior.
- `mask`: Original alpha mask for transparent mode, or the final alpha after image/color compositing.

Notes:

- In `transparent` mode, fully transparent pixels have RGB cleared to zero so hidden background pixels are not preserved.
- In `image` and `color` modes, the added background is clipped by `background_mask`. Pixels outside that mask remain transparent. If no mask is connected, the background fills the whole canvas behind transparent pixels.
- In `image` mode, if no background image is connected, the node falls back to transparent output.

## Notes

- SAM3 is a segmentation model, not a dedicated face detector. For unusual poses or stylized images, adjust the crop size or prompt.
- `feather_px` affects the square/circle crop mask only. It does not blur or stretch the foreground subject edge.
- Visible alpha-edge pixels are lightly decontaminated from nearby interior pixels to reduce background fringes when cutout is enabled, and when transparent inputs are used with cutout disabled.
- Square and circle crops with feathering are slightly inset so the feather finishes inside the output canvas instead of being clipped by the image boundary.
- For the crop node, transparent PNG inputs should connect the Load Image `mask` output to `source_transparency_mask` when the IMAGE socket does not carry alpha.
- Transparent RGB pixels are cleaned before output, and RGBA resize uses alpha-aware interpolation to avoid hidden background colors bleeding into transparent edges.

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
- 提供最终背景节点：不透明图像保持原样；透明图像可保持透明、添加上传背景图，或添加拾色器选择的纯色背景。
- 提供通用批量图片加载节点，支持多图上传、文件夹路径读取和标准 ComfyUI `IMAGE`/`MASK` batch 输出。
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

仓库也不会上传本地工作流和 Codex 维护临时文件：`workflows/` 和 `codex_maintenance_guide/`。

首次使用时，节点可以从 Hugging Face 自动下载所需文件：

- BiRefNet 文件来自 `1038lab/BiRefNet`
- SAM3 运行源码、权重和词表优先来自 `facebook/sam3`，并使用 `AB498/sam3` 作为备用来源

SAM3 运行源码会跳过大型模型权重文件复制；权重和词表会按固定路径单独下载。

如果需要离线安装，请手动放置模型文件：

```text
ComfyUI/custom_nodes/ComfyUI-Zcut/models/BiRefNet/*.safetensors
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/__init__.py
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/model_builder.py
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/model/sam3_image_processor.py
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/sam3.pt
ComfyUI/custom_nodes/ComfyUI-Zcut/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

如果使用私有或镜像 Hugging Face 仓库，请在启动 ComfyUI 前设置环境变量：

```bash
ZCUT_BIREFNET_HF_REPO=1038lab/BiRefNet
ZCUT_SAM3_HF_REPOS=facebook/sam3,AB498/sam3
```

## 节点说明

### Zcut Batch Load Images

`Zcut Batch Load Images` 是一个通用批量图片加载节点。它虽然放在 Zcut 插件里，并且沿用 Zcut 的命名方式，但输出的是 ComfyUI 标准的 `IMAGE` 和 `MASK` 类型，所以可以连接到其他 ComfyUI 原生节点或第三方插件节点使用。

主要功能：

- 支持一次选择多张图片上传。
- 支持输入本机文件夹路径，从文件夹中批量读取图片。
- 文件夹路径模式会自动忽略非图片文件。
- 可以把多张图片整理成一个标准图片 batch 输出。
- 支持处理图片尺寸不一致的情况，避免 batch 拼接失败。

输入参数：

- `upload_mode`：加载模式。
  - `multi_upload`：多图上传模式。点击节点底部的 `choose images` 按钮后，可以在电脑文件夹中一次选择多张图片上传。
  - `folder_path`：文件夹路径模式。手动输入本机文件夹路径，节点会从该文件夹中读取图片。
- `uploaded_images`：多图上传模式下保存上传图片信息的隐藏数据。正常使用时不需要手动修改，上传后节点会自动写入。
- `folder_path`：文件夹路径。只有当 `upload_mode` 选择 `folder_path` 时使用。这里填写要批量读取图片的本机文件夹路径。
- `recursive`：是否递归读取子文件夹。
  - `false`：只读取当前文件夹这一层。
  - `true`：当前文件夹和它下面的子文件夹都会一起扫描。
- `sort_order`：图片排序方式。
  - `name`：按文件名排序。
  - `modified_time`：按文件修改时间排序。
- `size_mode`：当多张图片尺寸不一致时的处理方式。
  - `pad_to_largest`：默认推荐方式。把所有图片居中放到最大宽高的透明画布中，不会拉伸原图。
  - `resize_to_first`：把所有图片缩放到第一张图片的尺寸。
  - `skip_mismatched`：只保留和第一张图片尺寸一致的图片，跳过尺寸不同的图片。
- `image_load_limit`：最多加载多少张图片。
  - `0`：不限制数量。
  - 例如设置为 `20`，就只加载排序后的前 20 张图片。
- `choose images`：多图上传按钮。点击后可以从电脑中批量选择图片，上传信息会自动保存到节点内部隐藏数据中。

输出：

- `images`：标准 ComfyUI `IMAGE` batch，可连接到其他图片处理节点。
- `masks`：标准 ComfyUI `MASK` batch。图片透明区域会转换为 mask。
- `file_paths`：实际加载的图片路径或上传图片引用，每张图片一行。
- `count`：本次实际加载进 batch 的图片数量。

使用建议：

- 如果只是临时选择几张图片处理，使用 `multi_upload` 更方便。
- 如果要批量处理一个固定文件夹中的大量图片，使用 `folder_path` 更合适。
- 如果图片尺寸不同，并且不想改变原图比例，建议使用默认的 `pad_to_largest`。
- 如果后续节点要求所有图片严格同尺寸，也可以使用 `resize_to_first`。

### Zcut BiRefNet SAM3 Face Crop

生成透明背景的人像/头像裁切图。

输入：

- `image`：ComfyUI 图像输入。
- `birefnet_model`：BiRefNet 模型版本。
- `source_transparency_mask`：可选的 ComfyUI 透明度 mask，例如 Load Image 的 `mask` 输出。当 IMAGE 输入不携带 alpha 时，透明 PNG 建议连接这个输入。启用抠图时，最终 mask 仍会尊重这份 alpha。
- `crop_width`、`crop_height`：输出裁切尺寸，单位为像素。
- `shape`：`square` 保留矩形裁切；`circle` 应用圆形 alpha 蒙版。
- `feather_edges`：启用方形/圆形裁切边缘羽化。
- `feather_px`：裁切边缘羽化半径，单位为像素。不会羽化人物边缘。方形和圆形裁切会自动内缩，确保羽化在画布边缘前完成。
- `device`：Auto、GPU 或 CPU。
- `birefnet_threshold`：前景 mask 阈值。
- `sam3_prompt`：用于 SAM3 定位脸部/头部的逗号分隔提示词。
- `sam3_confidence`：SAM3 置信度阈值。
- `enable_cutout`：是否启用 BiRefNet 前景抠图。关闭时会跳过 BiRefNet，保留原图内容；如果输入图带 alpha 通道，会保留原透明背景，并继续应用裁切形状蒙版。

输出：

- `cropped_image`：带透明背景的 RGBA 图像。
- `crop_mask`：最终 alpha mask。
- `birefnet_mask`：启用抠图时为原始 BiRefNet 前景 mask；关闭抠图时为叠加裁切形状前使用的基础 alpha mask。
- `shape_mask`：方形/圆形裁切形状 mask，包含同一组羽化设置。添加背景时请连接到 `Zcut Add Background` 的 `background_mask`。

### Zcut Resize Output

在裁切完成后调整最终头像输出尺寸。

输入：

- `image`：裁切后的 RGBA 图像。
- `resize_output`：是否启用最终画布尺寸调整。关闭时会直接透传图像和 mask，不改变尺寸。
- `output_width`、`output_height`：最终画布尺寸，单位为像素。
- `resample_method`：缩放重采样方式。`auto` 在缩小时使用 `area`，放大时使用 `lanczos`；也可以手动选择 `nearest-exact`、`bilinear`、`area`、`bicubic` 或 `lanczos`。
- `mask`：可选，同步调整的 mask。如果不连接，会使用图像 alpha 通道。

输出：

- `resized_image`：等比缩放并居中放入透明画布的图像。
- `resized_mask`：使用相同几何变换后的 mask。

### Zcut Image Upscale

使用 ComfyUI `models/upscale_models` 中的模型对图像进行放大或尺寸调整。

输入：

- `image`：ComfyUI 图像输入。
- `upscale_model`：来自 ComfyUI `models/upscale_models` 的放大模型名称。工作流保存的是相对模型名，不是绝对路径。
- `mode`：`rescale` 按系数缩放；`resize` 按目标尺寸缩放。
- `rescale_factor`：`rescale` 模式使用的缩放系数。
- `resize_mode`：`resize` 模式使用，可选 `longest_side`、`width`、`height`。
- `resize_size`：目标最长边、宽度或高度。
- `resample_method`：最终重采样方法。
- `supersample`：启用时，模型会先放大到不小于目标尺寸，再重采样到目标尺寸。
- `round_to_multiple`：最终宽高舍入到指定倍数。设为 `1` 表示不对齐。
输出：

- `upscaled_image`：放大后的 RGBA 图像。RGB 输入会按不透明处理；RGBA 输入会保留 alpha。
- `upscaled_mask`：使用相同几何变换后的 alpha mask。

说明：

- 目标尺寸大于输入时才会调用放大模型；目标尺寸更小时使用插值缩小。
- 透明输出会在模型放大和最终缩放后强制清空 alpha 全透明位置的 RGB 像素。
- 如需把结果放入最终透明画布且不改变宽高比例，请使用 `Zcut Resize Output`。

### Zcut Add Background

只在输入图像存在透明像素时添加背景。如果图像完全不透明，节点会直接输出原图，不做处理。

输入：

- `image`：最终图像输入。
- `background_mode`：`transparent` 保持透明背景，`image` 合成到上传背景图，`color` 合成到拾色器选择的纯色背景。
- `background_color`：纯色背景颜色，使用 ComfyUI 原生拾色器和吸管。
- `background_fit`：上传背景图适配当前图像尺寸的方式：`cover` 自动缩放并裁切填满，`stretch` 拉伸到完全相同尺寸。
- `background_mask`：可选，用于裁切图片/纯色背景的 mask。请连接裁切节点输出的 `shape_mask`，复用上一步的方形/圆形和羽化设置。
- `background_image`：可选上传背景图，在 `background_mode` 为 `image` 时使用。

输出：

- `image`：不透明输入时为原图；透明输入时为按所选背景模式处理后的 RGBA 图像。
- `mask`：透明模式下为原 alpha mask；图片/纯色背景合成后为最终 alpha。

说明：

- `transparent` 模式会清空全透明像素位置的 RGB，确保透明背景里没有隐藏像素。
- `image` 和 `color` 模式下，添加的背景会被 `background_mask` 裁切，mask 外仍保持透明；没有连接 mask 时，背景会填满整张画布的透明区域。
- `image` 模式没有连接背景图时，会回退为透明输出。

## 注意事项

- SAM3 是分割模型，不是专用人脸检测器。遇到特殊姿态或风格化图片时，可以调整裁切尺寸或提示词。
- `feather_px` 只影响方形/圆形裁切蒙版，不会模糊或拉伸人物边缘。
- 启用抠图时，以及透明图关闭抠图时，会用人物内部相邻颜色轻量清理 alpha 边界像素，减少背景色造成的边缘污染。
- 开启羽化的方形和圆形裁切会略微内缩，避免羽化还没结束就被图像边界截断。
- 对裁切节点，如果透明 PNG 的 IMAGE 输入不携带 alpha，请把 Load Image 的 `mask` 输出连接到 `source_transparency_mask`。
- 输出前会清理透明区域 RGB，RGBA 缩放也会使用 alpha 感知插值，避免隐藏背景颜色污染透明边缘。

## 致谢

特别感谢以下模型的作者和维护者：

- BiRefNet by 1038lab: https://huggingface.co/1038lab/BiRefNet
- SAM3 by Meta AI: https://huggingface.co/facebook/sam3
- 本插件使用的 SAM3 备用镜像: https://huggingface.co/AB498/sam3/tree/main

本仓库只包含插件代码。下载或使用模型权重时，请遵守对应模型的许可证和使用条款。
