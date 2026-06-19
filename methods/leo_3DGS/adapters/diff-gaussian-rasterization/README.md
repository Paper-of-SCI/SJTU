# Differential Gaussian Rasterization

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>


# 编译方法
```bash
cd /SJTU/methods/leo_3DGS/adapters/diff-gaussian-rasterization

# 先确认 Python / torch / CUDA：
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvcc --version
gcc --version

# 然后编译安装：
python -m pip install -v -e . --no-build-isolation


# 如果服务器也遇到 gcc 版本太新，比如 CUDA 报不支持当前 gcc，就用 gcc-12：
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
python -m pip install -v -e . --no-build-isolation

# 编译成功后测试：
python - <<'PY'
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print("diff-gaussian-rasterization import ok")
PY
```