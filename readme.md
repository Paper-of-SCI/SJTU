# gsplat编译需要的环境 ddd

## gsplat pip install gsplat 的包

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_INC="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
export NVIDIA_CU13="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13"

export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_INC:$NVIDIA_CU13/include:$CPATH"
export LIBRARY_PATH="$CUDA_LIB:$NVIDIA_CU13/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_LIB:$NVIDIA_CU13/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
```

## 来自GitHub的gsplat

### 先拉仓库

```bash
git clone --recursive https://github.com/nerfstudio-project/gsplat.git
```

> cd /home/leo/Projects/SJTU/methods/external/gsplat
> git submodule update --init --recursive

### 安装 gsplat editable

```bash
cd /home/leo/Projects/SJTU/methods/external/gsplat

BUILD_NO_CUDA=1 python -m pip install -e . --no-build-isolation
```

### 安装 example 依赖

```bash
python -m pip install \
  "imageio[ffmpeg]" tqdm tyro viser pyyaml torchmetrics==1.8.2 \
  opencv-python-headless Pillow piexif tensorboard matplotlib scipy scikit-learn splines pycolmap
```

### 装 nerfview

```bash
git -c http.version=HTTP/1.1 clone https://github.com/nerfstudio-project/nerfview /tmp/nerfview_manual
cd /tmp/nerfview_manual
git checkout 4538024fe0d15fd1a0e4d760f3695fc44ca72787
python -m pip install .
```

### 每次训练前设置环境变量

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_INC="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
export NVIDIA_CU13="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13"

export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_INC:$NVIDIA_CU13/include:$CPATH"
export LIBRARY_PATH="$CUDA_LIB:$NVIDIA_CU13/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_LIB:$NVIDIA_CU13/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.0"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

export BUILD_3DGUT=0
export BUILD_2DGS=0
export NUM_CHANNELS=3,4
# 这个可以不设置
# export MAX_JOBS=2

cd /home/leo/Projects/SJTU/methods/external/gsplat/examples

python leo_3dgs_trainer.py default \
  --data_dir /home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset/JapaneseGradens-RedSea/undistorted_pinhole \
  --data_factor 1 \
  --result_dir ./results/JapaneseGradens-RedSea-underwaterRasterizeFormula \
  --disable_viewer \
  --disable_video \
  --batch_size 1 \
  --eval_steps 1000 22000 29000 30000 \
  --save_steps 15000 30000 \
  --use_underwater_rasterize_formula \
  --depth-loss
```

### 把环境变量做出conda激活自动加载

```Shell
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

code "$CONDA_PREFIX/etc/conda/activate.d/gsplat_cuda.sh"
```

写进去

```
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_INC="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
export NVIDIA_CU13="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13"

export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_INC:$NVIDIA_CU13/include:$CPATH"
export LIBRARY_PATH="$CUDA_LIB:$NVIDIA_CU13/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_LIB:$NVIDIA_CU13/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.0"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

export BUILD_3DGUT=0
export BUILD_2DGS=0
export NUM_CHANNELS=3,4
```

然后重新开终端：

```bash
conda activate gsplat310
echo $BUILD_2DGS $BUILD_3DGUT $NUM_CHANNELS
```

> 应该输出：
> 0 0 3

### 单独构建编译

如果你之前有 build 缓存，可能还要清一下：

```bash
cd /home/leo/Projects/SJTU/methods/external/gsplat
rm -rf build
rm -rf ~/.cache/torch_extensions/py310_cu*
/home/leo/miniconda3/envs/gsplat310/bin/python -m pip install -e . --no-build-isolation --no-deps
```

### 评估

```bash
python simple_trainer.py default \
  --data_dir /home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset/JapaneseGradens-RedSea/undistorted_pinhole \
  --data_factor 1 \
  --result_dir ./results/curasao_eval \
  --disable_viewer \
  --disable_video \
  --ckpt ./results/curasao/ckpts/ckpt_9999_rank0.pt
```




# seasplat 编译cuda

## 环境

`Python`: 3.10
`PyTorch`: 2.11.0+cu128
`Torch CUDA`: 12.8
`NVCC`: 12.8
`GCC/G++`: 13.x
`GPU arch`: sm_120
`TORCH_CUDA_ARCH_LIST`: 12.0

### torch 安装指令

```bash
pip install \
  torch==2.11.0 \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

对应检查命令：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvcc --version
$CC --version
$CXX --version
```

理应看到：

> 2.11.0+cu128 12.8 NVIDIA GeForce RTX 5080
> Cuda compilation tools, release 12.8
> gcc/g++ 13.x

```bash
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate seasplat128
cd /home/leo/Projects/SJTU/methods/seasplat
```

先清掉容易污染编译的变量：

```bash
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CUDAHOSTCXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CUDAFLAGS
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64
export TORCH_CUDA_ARCH_LIST="12.0"
```

确认版本是对的：

```bash
which nvcc
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
$CC --version
$CXX --version
```

你要看到大概是：

> nvcc ... release 12.8
> torch ... +cu128 12.8 NVIDIA GeForce RTX 5080
> gcc/g++ 13.x

然后编译安装两个扩展：

```bash
python -m pip install submodules/diff-gaussian-rasterization --no-build-isolation --force-reinstall --no-cache-dir
python -m pip install submodules/simple-knn --no-build-isolation --force-reinstall --no-cache-dir
```

最后验证：

```bash
python -c "import torch; from simple_knn._C import distCUDA2; import diff_gaussian_rasterization; import gaussian_renderer; print('ok', torch.__version__, torch.version.cuda)"
```

> 如果输出 ok 2.11.0+cu128 12.8，说明编译和导入都没问题。

## 训练加评估

```bash
cd /home/leo/Projects/SJTU/methods/seasplat
conda activate seasplat128

python train.py \
-s /home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
--exp test_eval \
--eval \
--iterations 20000 \
--do_seathru --seathru_from_iter 10000
```

# 终端查看显卡信息

```bash
watch -n 1 nvidia-smi
```

---

# Conda

## Conda配置国内镜像

1. 直接执行

```bash
conda config --set show_channel_urls yes
```

2. 然后打开配置文件：

```bash
nano ~/.condarc
```

3. 把内容改成这样：

```text
channels:
  - defaults

show_channel_urls: true

default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

4. 保存后清缓存：

```bash
conda clean -i
```



## conda 创建、删除、退出、查看、冻结和解冻环境

### 创建

```bash
conda create -n <环境名> python=3.10 -y
```

### 删除

```bash
#先退出
conda deactivate

#再删除
conda env remove -n <环境名>
```

### 退出

```bash
conda deactivate
```

### 查看

```bash
conda env list
```

### 冻结和解冻环境

```bash
#先激活要冻结的环境
conda activate seasplat128

# 然后创建 frozen 标记：
touch "$CONDA_PREFIX/conda-meta/frozen"

# 确认看到环境名前面/旁边有 + 就是 frozen。：
conda env list

# 解冻当前环境
rm "$CONDA_PREFIX/conda-meta/frozen"

# 确认+ 消失就解冻了。：
conda env list
```

---

# 生成无畸变图片

单场景

```bash
SCENE=Curasao
ROOT=/home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset

colmap image_undistorter \
  --image_path "${ROOT}/${SCENE}/images_wb" \
  --input_path "${ROOT}/${SCENE}/sparse/0" \
  --output_path "${ROOT}/${SCENE}/undistorted_pinhole" \
  --output_type COLMAP \
  --max_image_size 2000
```

多场景

```bash
ROOT=/home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset

scenes=(
  Curasao
  IUI3-RedSea
  JapaneseGradens-RedSea
  Panama
)

for scene in "${scenes[@]}"; do
  echo "Undistorting ${scene}"

  colmap image_undistorter \
    --image_path "${ROOT}/${scene}/images_wb" \
    --input_path "${ROOT}/${scene}/sparse/0" \
    --output_path "${ROOT}/${scene}/undistorted_pinhole" \
    --output_type COLMAP \
    --max_image_size 2000

  mkdir -p "${ROOT}/${scene}/undistorted_pinhole/sparse/0"

  if ls "${ROOT}/${scene}/undistorted_pinhole/sparse/"*.bin >/dev/null 2>&1; then
    mv "${ROOT}/${scene}/undistorted_pinhole/sparse/"*.bin \
       "${ROOT}/${scene}/undistorted_pinhole/sparse/0/"
  fi
done
```


# Ubuntu

## Ubuntu设置国内镜像

1. 先查看源

```bash
grep -R "archive.ubuntu.com\|security.ubuntu.com" /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null
```

2. 找到修改镜像的文件

```bash
ls /etc/apt/sources.list.d
```

3. 打开要修改的文件

```bash
sudo nano /etc/apt/sources.list.d/ubuntu.sources
```

4. 修改镜像

`URIs: http://archive.ubuntu.com/ubuntu/`   $\to$ `URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ `

如果还有：`URIs: http://security.ubuntu.com/ubuntu/` $\to$  `URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu/`



---

# Pip

## 配置pip国内镜像

1. 执行

```bash
python -m pip install --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

2. 查看当前 pip 配置：

```bash
pip config list
```
