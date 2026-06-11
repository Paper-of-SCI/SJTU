# 写入 `构建渲染和loss思路.md` 的算法说明计划

## Summary

把文档写成一份可以直接照着实现的算法说明，重点说明：

- `scene Gaussians` 负责物体几何、颜色和纹理；
- `medium field` 负责水体介质，包括衰减、雾化、偏色、backscatter；
- 渲染输出必须拆成 `rgb_object`、`rgb_medium`、`pred_image`；
- loss 不只是让 `pred_image` 接近 GT，还要限制 medium 不要抢物体纹理表达权。

文档采用 WaterSplatting 现有接口命名：

- `medium_attn`：直接光衰减系数；
- `medium_bs`：backscatter 散射系数；
- `medium_rgb`：水体散射颜色；
- `rgb_object`：物体直接光项；
- `rgb_medium`：水体散射项；
- `pred_image`：最终渲染图像。

## Key Changes

文档内容分成 7 个部分。

1. 总体目标

说明最终模型不是单纯追求 PSNR，而是让最终图像由物体项和水体项共同解释：

$$
I(u)
=
I_{\mathrm{obj}}(u)
+
I_{\mathrm{med}}(u)
$$

其中：

- \(I(u)\)：最终渲染图像，对应代码里的 `pred_image`；
- \(I_{\mathrm{obj}}(u)\)：物体直接光，对应 `rgb_object`；
- \(I_{\mathrm{med}}(u)\)：水体 backscatter / haze，对应 `rgb_medium`。

训练目标是：

$$
I(u)\approx I_{gt}(u)
$$

但同时要求：

- 物体纹理主要由 \(I_{\mathrm{obj}}\) 解释；
- 水体雾化、偏色、散射主要由 \(I_{\mathrm{med}}\) 解释；
- medium field 是低频、平滑、低容量的 3D 水体场。

2. 渲染公式

写明 3DGS 原始物体颜色：

$$
J(u)
=
\sum_i
T_i^{gs}(u)\alpha_i(u)c_i
$$

其中：

- \(i\)：当前像素上按深度排序的第 \(i\) 个 scene Gaussian；
- \(T_i^{gs}(u)\)：第 \(i\) 个 Gaussian 前方的 scene visibility；
- \(\alpha_i(u)\)：第 \(i\) 个 Gaussian 对像素 \(u\) 的 opacity；
- \(c_i\)：第 \(i\) 个 Gaussian 的颜色。

medium field 表示水体消光密度：

$$
\boldsymbol{\beta}(x)
\in \mathbb{R}_+^3
$$

从相机到深度 \(t\) 的水体透射率：

$$
\boldsymbol{T}_m(u,t)
=
\exp
\left(
-\int_0^t
\boldsymbol{\beta}(r_u(s))ds
\right)
$$

最终物体直接光项：

$$
I_{\mathrm{obj}}(u)
=
\sum_i
T_i^{gs}(u)\alpha_i(u)c_i
\odot
\boldsymbol{T}_m(u,t_i)
$$

简化 backscatter 项：

$$
I_{\mathrm{med}}(u)
=
b_\infty
\odot
\left(
1-\boldsymbol{T}_m(u,D_{\mathrm{vis}})
\right)
$$

最终渲染：

$$
I(u)
=
I_{\mathrm{obj}}(u)
+
I_{\mathrm{med}}(u)
$$

3. WaterSplatting 接口版本

为了贴近现有代码，文档明确第一版不直接改 CUDA 的逐段采样逻辑，而是把 3D medium field 沿 ray 积分成 WaterSplatting 需要的三张 per-pixel 图。

对每个像素 \(u\)，先采样 ray：

$$
r_u(s)=o+sd_u
$$

选择积分终点：

$$
D_{\mathrm{ref}}(u)
=
D_{\mathrm{vis}}(u)
$$

其中 \(D_{\mathrm{vis}}\) 优先使用渲染深度 `depth`，没有可靠前景时使用 far plane。

计算平均 attenuation：

$$
\overline{\boldsymbol{\beta}}(u)
=
\frac{1}
{\max(D_{\mathrm{ref}}(u),\epsilon)}
\int_0^{D_{\mathrm{ref}}(u)}
\boldsymbol{\beta}(r_u(s))ds
$$

对应代码变量：

$$
\mathrm{medium\_attn}(u)
=
\overline{\boldsymbol{\beta}}(u)
$$

backscatter density 使用比例约束：

$$
\boldsymbol{\sigma}_b(x)
=
\boldsymbol{\rho}_b(x)
\odot
\boldsymbol{\beta}(x),
\quad
0\leq \boldsymbol{\rho}_b(x)\leq 1
$$

计算平均 scattering：

$$
\overline{\boldsymbol{\sigma}}_b(u)
=
\frac{1}
{\max(D_{\mathrm{ref}}(u),\epsilon)}
\int_0^{D_{\mathrm{ref}}(u)}
\boldsymbol{\sigma}_b(r_u(s))ds
$$

对应代码变量：

$$
\mathrm{medium\_bs}(u)
=
\overline{\boldsymbol{\sigma}}_b(u)
$$

medium 颜色：

$$
\mathrm{medium\_rgb}(u)
=
b_\infty
$$

其中 \(b_\infty\) 是可学习 RGB，使用 sigmoid 或 clamp 限制到 \([0,1]\)。

4. Loss 构建

总 loss 写成：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{medium}\mathcal{L}_{medium}
+
\lambda_{scatter}\mathcal{L}_{scatter}
+
\lambda_{decor}\mathcal{L}_{decor}
$$

第一项是基础重建：

$$
\mathcal{L}_{rgb}
=
\|I-I_{gt}\|_1
+
\lambda_{ssim}
\left(
1-\mathrm{SSIM}(I,I_{gt})
\right)
$$

其中：

- \(I\)：`pred_image`；
- \(I_{gt}\)：GT 图像。

medium 平滑和低容量约束：

$$
\mathcal{L}_{medium}
=
\frac{1}{|\mathcal{S}|}
\sum_{x\in\mathcal{S}}
\|\nabla \boldsymbol{\beta}(x)\|_2^2
+
\lambda_\beta
\frac{1}{|\mathcal{S}|}
\sum_{x\in\mathcal{S}}
\|\boldsymbol{\beta}(x)\|_1
$$

其中 \(\mathcal{S}\) 是从场景 bbox 或训练 ray 上采样出的 3D 点集合。

scatter 合法性约束：

$$
\mathcal{L}_{scatter}
=
\frac{1}{|\mathcal{S}|}
\sum_{x\in\mathcal{S}}
\left\|
\max
\left(
0,
\boldsymbol{\sigma}_b(x)-\boldsymbol{\beta}(x)
\right)
\right\|_1
$$

如果实现时直接使用：

$$
\boldsymbol{\sigma}_b(x)
=
\mathrm{sigmoid}(g_\theta(x))
\odot
\boldsymbol{\beta}(x)
$$

则 \(\mathcal{L}_{scatter}\) 可以不用显式加入。

5. Medium 不抢纹理 loss

定义水体影响图：

$$
M(u)
=
I(u)-J(u)
$$

实现中建议使用：

$$
M(u)
=
\mathrm{rgb\_medium}(u)
+
\left(
\mathrm{rgb\_object}(u)-J(u)
\right)
$$

如果没有单独输出无水 \(J(u)\)，第一版可以近似为：

$$
M(u)
=
\mathrm{rgb\_medium}(u)
$$

计算图像梯度：

$$
G_M(u)
=
\|\nabla M(u)\|_1
$$

$$
G_J(u)
=
\|\nabla J(u)\|_1
$$

如果没有无水 \(J(u)\)，第一版用 `rgb_object` 近似：

$$
G_J(u)
=
\|\nabla \mathrm{rgb\_object}(u)\|_1
$$

去相关 loss：

$$
\mathcal{L}_{decor}
=
\left|
\mathrm{Corr}(G_M,G_J)
\right|
$$

其中相关系数按当前图像所有有效像素计算：

$$
\mathrm{Corr}(a,b)
=
\frac{
\sum_u
(a(u)-\bar a)(b(u)-\bar b)
}{
\sqrt{
\sum_u(a(u)-\bar a)^2
}
\sqrt{
\sum_u(b(u)-\bar b)^2
}
+\epsilon
}
$$

这个 loss 的含义要写清楚：

- 物体边缘和纹理会让 \(G_J\) 变大；
- 水体项 \(M\) 不应该在同一位置同步产生强边缘；
- 如果二者高度相关，说明 medium field 正在拟合物体纹理；
- 所以最小化 \(\mathcal{L}_{decor}\)。

6. 可选跨视角 loss

作为第二阶段或增强版写入文档，不作为第一版必须实现项。

同一个 3D 点在两个视角 \(v_1,v_2\) 中可见时，反推去水体后的颜色：

$$
\hat J_v(u)
=
\frac{
I_{gt,v}(u)-I_{\mathrm{med},v}(u)
}{
T_{m,v}(u,D_v)+\epsilon
}
$$

跨视角一致性：

$$
\mathcal{L}_{view}
=
\sum_{(v_1,u_1),(v_2,u_2)}
\left\|
\hat J_{v_1}(u_1)
-
\hat J_{v_2}(u_2)
\right\|_1
$$

文档注明这个 loss 需要跨视角对应关系，第一版可以不实现；等已有 depth、camera pose、可见性判断稳定后再加入。

7. 第一版推荐实现顺序

文档最后给出明确落地顺序：

1. 保持现有 WaterSplatting rasterizer 主路径不变。
2. 新增 3D medium field，用它生成 `medium_attn`、`medium_bs`、`medium_rgb`。
3. 渲染输出继续使用 `rgb_object + rgb_medium = pred_image`。
4. 第一版 loss 使用：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{medium}\mathcal{L}_{medium}
+
\lambda_{decor}\mathcal{L}_{decor}
$$

5. 如果 \(\boldsymbol{\sigma}_b\) 没有用比例参数化，再加入 \(\mathcal{L}_{scatter}\)。
6. 等训练稳定后，再加入 \(\mathcal{L}_{view}\)。

## Test Plan

文档里要写清楚实现后的验证方式：

- 关闭 medium 时，`medium_attn=0`、`medium_bs=0`、`rgb_medium=0`，渲染应退化为普通 3DGS / 物体项。
- 开启 medium 后，应满足：

$$
\mathrm{pred\_image}
=
\mathrm{rgb\_object}
+
\mathrm{rgb\_medium}
$$

- `medium_attn`、`medium_bs` 应为非负。
- `medium_rgb` 应在 \([0,1]\)。
- `rgb_medium` 应更平滑，不能明显复制物体纹理边缘。
- 记录并可视化：

$$
\mathrm{medium\_attn},
\quad
\mathrm{medium\_bs},
\quad
\mathrm{medium\_rgb},
\quad
\mathrm{rgb\_object},
\quad
\mathrm{rgb\_medium}
$$

## Assumptions

- `medium` 在文档中统一翻译为“水体介质”。
- 第一版以 WaterSplatting 现有 `medium_attn`、`medium_bs`、`medium_rgb` 接口为主，不要求先改 CUDA kernel。
- 第一版不强制实现跨视角 \(\mathcal{L}_{view}\)，因为它需要额外的跨视角匹配和可见性判断。
- 最推荐的第一版核心创新 loss 是：

$$
\mathcal{L}_{decor}
$$

它直接约束 medium field 不要拟合物体纹理。
