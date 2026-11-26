# BuildOne

用同一刚体的一张或多张照片，或一段视频，生成可编辑的 build123d 模型、带碰撞体的 USD 资产和离线 3D 预览。

## 首次安装

需要 Linux x86-64、支持 BF16 的 NVIDIA GPU、兼容 CUDA 12.9 的驱动，以及 Bash、Git、patch、curl 和 OpenGL 运行库（Ubuntu 的 `libgl1`）。安装脚本会准备 uv、Python 3.11 和锁定的依赖。首次安装需要联网，运行时 Codex 也需要联网。

1. 登录 Hugging Face，在 SAM 3.1 页面申请访问权限。获批后，安装时可用同一账号通过浏览器登录，也可粘贴该账号的只读 token。脚本会复用已有登录。非交互安装可设置 `HF_TOKEN`。[SAM 3.1 访问申请](https://huggingface.co/facebook/sam3.1) · [创建 token](https://huggingface.co/settings/tokens)
2. 克隆并运行安装脚本。

   ```sh
   git clone https://github.com/klee101/buildone.git
   cd buildone
   ./setup.sh
   ```

   脚本下载 `facebook/sam3.1` 的 `sam3.1_multiplex.pt`（约 3.5 GB）到 `models/sam3.1/`，并复用当前 Codex 登录和模型设置。首次使用 Codex 时，按终端提示打开链接登录，也可在安装前设置 `OPENAI_API_KEY`。使用的 Codex 模型需支持图片输入。

已有 SAM 权重可跳过 Hugging Face 登录。在运行 `./setup.sh` 前，执行 `cp docs/example.env .env`，按注释填写权重路径。也可以在这里指定 Codex 模型。`setup.sh` 和 `run.sh` 都会读取这个文件。

## 运行

用仓库自带的三张照片生成第一个资产。

```sh
./run.sh --input data/*.jpg --size-prompt "物体从上到下约 17 厘米"
```

换成自己的照片或视频时，只需替换 `--input` 后的文件。尺寸提示可省略。终端会打印本次输出目录，其中的 `model.py` 可继续编辑，`asset.usd` 可导入 Isaac Sim，`viewer.html` 可直接用支持 WebGL 2 的浏览器打开。

如需额外生成四张参考视图，运行以下两条命令。安装脚本会将 Qwen 的 Diffusers 权重下载到 `models/Qwen-Image-2.1-Turbo/`（约 32.5 GB，推理还需容纳权重和中间张量的显存）。[Qwen 模型文件](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/main)

```sh
./setup.sh --qwen-views
./run.sh --input data/*.jpg --qwen-views
```
