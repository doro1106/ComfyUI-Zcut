# ComfyUI-Zcut 后续开发和维护操作手册

这份文档写给不熟悉电脑和代码操作的用户。你可以把它当成检查清单：以后换到另一台电脑，想继续用 Codex 修改、维护、上传这个插件时，按顺序做即可。

当前插件 GitHub 地址：

```text
https://github.com/doro1106/ComfyUI-Zcut
```

## 1. 先理解几个概念

### 1.1 GitHub 是什么

GitHub 是保存代码的云端仓库。你的插件代码已经上传到 GitHub：

```text
https://github.com/doro1106/ComfyUI-Zcut
```

以后其他电脑不需要从旧电脑复制代码，可以直接从 GitHub 下载。

### 1.2 Git 是什么

Git 是管理代码版本的工具。它可以记录你每次修改了什么，也可以把本地修改上传到 GitHub。

常见命令：

```bash
git clone 仓库地址
git status
git add 文件名
git commit -m "说明文字"
git push
git pull
```

### 1.3 Codex 是什么

Codex 是帮你读代码、改代码、测试代码的开发助手。它需要在插件目录里运行，这样它才能看到并修改插件文件。

### 1.4 gh CLI 是什么

`gh` 是 GitHub 官方命令行工具。它可以登录 GitHub、创建 release、查看仓库等。

你不一定每次都需要用它，但建议安装。

## 2. 新电脑需要准备什么

新电脑上建议准备这些东西：

- ComfyUI
- Git
- Node.js，主要用于安装/运行 Codex CLI
- Codex CLI
- GitHub CLI，也就是 `gh`
- 一个能访问 GitHub 的网络环境
- 如果不能访问 Hugging Face，还需要手动准备模型文件

## 3. 在新电脑安装 Git

### 3.1 下载 Git

打开浏览器，访问：

```text
https://git-scm.com/download/win
```

下载 Windows 版本并安装。

### 3.2 安装时怎么选

如果看不懂安装选项，基本一路点击 `Next` 即可。

### 3.3 检查 Git 是否安装成功

打开 PowerShell，输入：

```bash
git --version
```

如果看到类似下面的内容，说明安装成功：

```text
git version 2.x.x
```

## 4. 在新电脑安装 Node.js

### 4.1 下载 Node.js

打开浏览器，访问：

```text
https://nodejs.org/
```

下载 LTS 版本并安装。

### 4.2 检查 Node.js 是否安装成功

打开 PowerShell，输入：

```bash
node --version
npm --version
```

如果都能显示版本号，说明安装成功。

## 5. 安装 Codex CLI

在 PowerShell 中运行：

```bash
npm install -g @openai/codex
```

安装完成后检查：

```bash
codex --version
```

如果显示版本号，说明 Codex CLI 安装成功。

然后登录 Codex：

```bash
codex --login
```

按提示完成登录。

## 6. 安装 GitHub CLI

### 6.1 推荐方式：使用 winget

在 PowerShell 中运行：

```bash
winget install --id GitHub.cli
```

安装完成后检查：

```bash
gh --version
```

### 6.2 如果 winget 不可用

打开浏览器访问 GitHub CLI 发布页：

```text
https://github.com/cli/cli/releases
```

下载 Windows 版本安装包或 zip。安装完成后，重新打开 PowerShell，再运行：

```bash
gh --version
```

### 6.3 登录 GitHub CLI

运行：

```bash
gh auth login
```

按这个顺序选择：

```text
GitHub.com
HTTPS
Y
Login with a web browser
```

如果出现 one-time code：

1. 复制终端里显示的 code。
2. 按 Enter 打开浏览器。
3. 在 GitHub 网页里输入 code。
4. 授权 GitHub CLI。
5. 回到终端，等它显示登录成功。

检查登录状态：

```bash
gh auth status
```

如果看到 `Logged in to github.com account doro1106`，说明成功。

## 7. 在新电脑下载插件代码

假设你的 ComfyUI 在：

```text
D:\ComfyUI
```

如果你的 ComfyUI 路径不同，请把下面命令里的路径换成你自己的。

打开 PowerShell：

```bash
cd D:\ComfyUI\custom_nodes
git clone https://github.com/doro1106/ComfyUI-Zcut.git
```

下载完成后，会出现这个目录：

```text
D:\ComfyUI\custom_nodes\ComfyUI-Zcut
```

进入插件目录：

```bash
cd D:\ComfyUI\custom_nodes\ComfyUI-Zcut
```

检查文件：

```bash
dir
```

应该能看到：

```text
nodes.py
README.md
requirements.txt
install.py
__init__.py
LICENSE
.gitignore
```

## 8. 处理模型文件

这个插件的 GitHub 仓库不包含 `models` 文件夹，因为模型文件很大。

你有两种选择。

### 8.1 方式一：让插件自动下载模型

如果新电脑可以访问 Hugging Face，直接启动 ComfyUI，第一次使用节点时插件会尝试自动下载模型。

自动下载来源：

```text
BiRefNet: https://huggingface.co/1038lab/BiRefNet
SAM3: https://huggingface.co/facebook/sam3
SAM3 fallback: https://huggingface.co/AB498/sam3/tree/main
```

### 8.2 方式二：手动复制模型文件

如果新电脑不能访问 Hugging Face，就从旧电脑复制 `models` 文件夹。

旧电脑位置可能是：

```text
G:\AI\Zcut\models
```

新电脑目标位置应该是：

```text
D:\ComfyUI\custom_nodes\ComfyUI-Zcut\models
```

最终结构应该类似：

```text
ComfyUI-Zcut
  models
    BiRefNet
      *.safetensors
    sam3
      sam3.pt
      assets
        bpe_simple_vocab_16e6.txt.gz
```

## 9. 安装插件依赖

进入插件目录：

```bash
cd D:\ComfyUI\custom_nodes\ComfyUI-Zcut
```

安装依赖：

```bash
pip install -r requirements.txt
```

注意：如果你使用的是 ComfyUI Portable，普通 `pip` 可能不是 ComfyUI 的 Python。遇到这种情况，需要用 ComfyUI 自带的 Python。

常见 Portable 示例：

```bash
D:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt
```

路径要按你自己的 ComfyUI 实际位置调整。

## 10. 启动 ComfyUI 并检查插件

重启 ComfyUI。

在 ComfyUI 里搜索节点：

```text
Zcut
```

应该能看到：

```text
Zcut BiRefNet SAM3 Face Crop
Zcut Resize Output
Zcut Image Upscale
Zcut Add Background
```

如果看不到：

1. 看 ComfyUI 启动窗口有没有红色报错。
2. 检查插件是否放在 `custom_nodes` 目录。
3. 检查依赖是否安装成功。
4. 检查模型文件是否存在，或网络是否能下载模型。

## 11. 在新电脑用 Codex 继续维护插件

进入插件目录：

```bash
cd D:\ComfyUI\custom_nodes\ComfyUI-Zcut
```

启动 Codex：

```bash
codex
```

以后你可以这样告诉 Codex：

```text
请帮我修改这个 ComfyUI 插件，新增一个节点……
```

或者：

```text
请检查 nodes.py，看看为什么这个节点输出报错……
```

重要提醒：一定要在插件目录里启动 Codex。也就是目录里能看到 `nodes.py` 和 `README.md` 的地方。

## 12. 修改代码前先同步 GitHub 最新版本

每次开始开发前，先进入插件目录：

```bash
cd D:\ComfyUI\custom_nodes\ComfyUI-Zcut
```

拉取 GitHub 最新代码：

```bash
git pull
```

这样可以避免你在旧代码上继续修改。

## 13. 让 Codex 修改代码时怎么说

你可以用比较明确的说法，效果会更好。

示例一：

```text
请新增一个 ComfyUI 节点，用来把输出图像转换成指定格式。修改后运行 python -m py_compile nodes.py 检查语法。
```

示例二：

```text
节点运行报错，错误信息是……请定位原因并修复。不要修改 models 文件夹。
```

示例三：

```text
请更新 README，补充新节点的中文和英文说明。
```

## 14. 每次修改后要做的检查

至少运行：

```bash
python -m py_compile nodes.py
```

如果没有输出，通常说明语法检查通过。

然后重启 ComfyUI，实际跑一遍节点。

如果修改了 README，可以打开 GitHub 页面看排版是否正常。

## 15. 查看自己改了哪些文件

进入插件目录，运行：

```bash
git status
```

常见结果：

```text
modified: nodes.py
modified: README.md
```

表示这些文件被修改了。

查看具体修改：

```bash
git diff
```

如果看不懂，可以让 Codex 解释：

```text
请帮我解释当前 git diff 里改了什么。
```

## 16. 把修改保存成一次 Git 提交

确认修改没问题后，运行：

```bash
git add nodes.py README.md
git commit -m "Update Zcut node behavior"
```

如果你还修改了其他文件，就把文件名也加进去。

也可以一次添加所有已跟踪文件：

```bash
git add .
git commit -m "Update plugin"
```

注意：`models/`、`workflows/` 和 `codex_maintenance_guide/` 已经被 `.gitignore` 排除，不会上传。

## 17. 把修改上传到 GitHub

提交后运行：

```bash
git push
```

上传完成后，打开：

```text
https://github.com/doro1106/ComfyUI-Zcut
```

检查代码是否更新。

## 18. 创建新版本 Release

如果你做了一次比较重要的更新，可以创建新版本。

版本号示例：

```text
v0.1.1
v0.2.0
v1.0.0
```

一般规则：

- 小修小补：`v0.1.1`
- 新增功能：`v0.2.0`
- 重大稳定版本：`v1.0.0`

创建 tag：

```bash
git tag -a v0.1.1 -m "Zcut ComfyUI Nodes v0.1.1"
git push origin v0.1.1
```

创建 GitHub Release：

```bash
gh release create v0.1.1 --title "Zcut ComfyUI Nodes v0.1.1" --notes "Describe what changed in this release."
```

你也可以让 Codex 帮你生成 release 文案：

```text
请根据最近一次 git diff，帮我生成中英双语 release notes。
```

## 19. 在另一台电脑同步你的最新修改

如果另一台电脑已经 clone 过仓库，只需要：

```bash
cd D:\ComfyUI\custom_nodes\ComfyUI-Zcut
git pull
```

然后重启 ComfyUI。

如果修改涉及依赖，再运行：

```bash
pip install -r requirements.txt
```

## 20. 常见问题

### 20.1 PowerShell 提示找不到 git

说明 Git 没安装，或安装后没有重新打开 PowerShell。

解决：

1. 安装 Git。
2. 关闭 PowerShell。
3. 重新打开 PowerShell。
4. 再运行 `git --version`。

### 20.2 PowerShell 提示找不到 codex

说明 Codex CLI 没安装成功，或 Node.js/npm 没准备好。

检查：

```bash
npm --version
```

重新安装：

```bash
npm install -g @openai/codex
```

### 20.3 PowerShell 提示找不到 gh

说明 GitHub CLI 没安装，或 PATH 没刷新。

解决：

1. 重新打开 PowerShell。
2. 运行 `gh --version`。
3. 如果仍然找不到，重新安装 GitHub CLI。

### 20.4 git push 失败

先检查是否登录 GitHub CLI：

```bash
gh auth status
```

如果没登录：

```bash
gh auth login
```

再尝试：

```bash
git push
```

### 20.5 ComfyUI 看不到 Zcut 节点

检查这些点：

1. 插件是否在 `ComfyUI/custom_nodes/ComfyUI-Zcut`。
2. 是否重启了 ComfyUI。
3. ComfyUI 启动窗口是否有红色报错。
4. 是否安装了 `requirements.txt`。
5. 模型文件是否能下载或已手动放好。

### 20.6 不小心改坏了代码

如果还没有提交，可以查看改了什么：

```bash
git diff
```

如果想放弃某个文件的本地修改：

```bash
git restore nodes.py
```

注意：这会丢掉你对 `nodes.py` 的本地修改。执行前最好问 Codex：

```text
我想撤销 nodes.py 的修改，请先帮我确认 git diff 里有没有重要内容。
```

## 21. 推荐的日常维护流程

每次开发建议按这个顺序：

```text
1. 打开 PowerShell
2. cd 到 ComfyUI-Zcut 插件目录
3. git pull
4. codex
5. 让 Codex 修改或检查代码
6. python -m py_compile nodes.py
7. 重启 ComfyUI 实测
8. git status
9. git add ...
10. git commit -m "说明"
11. git push
```

## 22. 给 Codex 的推荐固定提醒

以后你让 Codex 修改这个插件时，可以在请求最后加上：

```text
请不要上传或修改 models、workflows、codex_maintenance_guide 文件夹。修改后请运行 python -m py_compile nodes.py。README 如果涉及节点行为变化，请同步更新中英双语说明。
```

这样能减少误操作。

## 23. 最重要的备份建议

GitHub 已经保存了插件代码，但没有保存模型文件。

所以你至少要备份：

```text
models/
```

建议把 `models` 文件夹单独复制到移动硬盘、网盘或另一台电脑。

代码可以从 GitHub 重新下载，模型文件如果网络不好，重新下载会比较麻烦。
