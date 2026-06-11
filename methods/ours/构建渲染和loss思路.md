# 3D Medium Field 渲染与 Loss 构建思路

本文只写算法思路，不写具体工程代码。目标是让后续实现时可以直接按照这里的变量、渲染输出和 loss 公式落地。

这里的 `medium` 统一表示“水体介质”，不是“中等”的意思。它负责解释水下图像里的衰减、雾化、偏色和 backscatter；`scene Gaussians` 负责解释物体几何、物体颜色、纹理和边缘。

## 1. 总体目标

最终渲染图像拆成两部分：

$$
I(u)
=
I_{\mathrm{obj}}(u)
+
I_{\mathrm{med}}(u)
$$

其中：

- \(u\)：图像像素位置。
- \(I(u)\)：最终渲染图像，对应代码里的 `pred_image`。
- \(I_{\mathrm{obj}}(u)\)：物体直接光项，对应代码里的 `rgb_object`。
- \(I_{\mathrm{med}}(u)\)：水体散射、雾化和偏色项，对应代码里的 `rgb_medium`。
- \(I_{gt}(u)\)：真实水下图像，也就是 GT image。

训练时基础目标是：

$$
I(u)
\approx
I_{gt}(u)
$$

但这个方法的重点不是单纯提高 PSNR，而是让“拟合 GT”这件事通过合理的物体-水体分解完成：

- 物体纹理、边缘、结构主要由 \(I_{\mathrm{obj}}\) 解释。
- 水体雾化、偏色、backscatter 主要由 \(I_{\mathrm{med}}\) 解释。
- `medium field` 应该是低频、平滑、低容量的 3D 水体场。
- `medium field` 不应该变成一个自由纹理场去复制物体细节。

第一版建议沿用 WaterSplatting 的输出命名：

- `medium_attn`：直接光 attenuation 系数。
- `medium_bs`：backscatter scattering 系数。
- `medium_rgb`：水体散射颜色。
- `rgb_object`：经过水体衰减后的物体直接光项。
- `rgb_medium`：水体散射项。
- `pred_image`：最终渲染图像。

需要始终满足：

$$
\mathrm{pred\_image}(u)
=
\mathrm{rgb\_object}(u)
+
\mathrm{rgb\_medium}(u)
$$

## 2. 基础 3DGS 物体渲染项

先定义不考虑水体时，原始 3DGS 的物体颜色：

$$
J(u)
=
\sum_i
T_i^{gs}(u)
\alpha_i(u)
c_i
$$

其中：

- \(i\)：当前像素上按深度从近到远排序的第 \(i\) 个 `scene Gaussian`。
- \(c_i\)：第 \(i\) 个 `scene Gaussian` 的颜色，可以来自 SH 或 RGB 参数。
- \(\alpha_i(u)\)：第 \(i\) 个 `scene Gaussian` 投影到像素 \(u\) 后的有效 opacity。
- \(T_i^{gs}(u)\)：第 \(i\) 个 Gaussian 前方的 scene visibility。

scene visibility 为：

$$
T_i^{gs}(u)
=
\prod_{j<i}
\left(
1-\alpha_j(u)
\right)
$$

因此 \(J(u)\) 表示“没有水体介质影响时”的物体渲染结果。后续如果实现里能输出无水颜色，就把它命名为 `rgb_clear_raw` 或 `J`；如果第一版没有这个输出，可以先用 `rgb_object` 近似参与部分 loss。

## 3. 3D Medium Field 表示

新增一个低容量 3D 水体场：

$$
\boldsymbol{\beta}(x)
\in
\mathbb{R}_+^3
$$

其中：

- \(x\in\mathbb{R}^3\)：3D 空间位置。
- \(\boldsymbol{\beta}(x)\)：RGB 三通道水体消光密度，也就是 extinction density。
- 数值越大，表示该位置水体越浑浊，直接光衰减越强。

可以用两种方式实现 \(\boldsymbol{\beta}(x)\)。

第一种是低容量 MLP：

$$
\boldsymbol{\beta}(x)
=
\mathrm{softplus}
\left(
f_\theta(x)
\right)
$$

第二种是少量 `medium Gaussians`：

$$
\boldsymbol{\beta}(x)
=
\sum_k
\boldsymbol{w}_k
G_k(x)
$$

其中：

$$
G_k(x)
=
\exp
\left(
-
\frac{1}{2}
(x-m_k)^T
\Sigma_k^{-1}
(x-m_k)
\right)
$$

其中：

- \(k\)：第 \(k\) 个 `medium Gaussian`。
- \(m_k\)：第 \(k\) 个 `medium Gaussian` 的 3D 中心。
- \(\Sigma_k\)：第 \(k\) 个 `medium Gaussian` 的 3D 协方差。
- \(\boldsymbol{w}_k\)：第 \(k\) 个 `medium Gaussian` 的 RGB 强度，要求非负。

第一版更建议用少量 `medium Gaussians` 或很小的 MLP，因为它天然限制了 medium 的表达能力，避免它抢物体纹理。

## 4. 路径积分透射率

像素 \(u\) 对应的 camera ray 写成：

$$
r_u(s)
=
o
+
s d_u
$$

其中：

- \(o\)：相机中心。
- \(d_u\)：像素 \(u\) 对应的单位 ray 方向。
- \(s\)：沿 ray 的路径长度。

从相机到深度 \(t\) 的水体透射率为：

$$
\boldsymbol{T}_m(u,t)
=
\exp
\left(
-
\int_0^t
\boldsymbol{\beta}
\left(
r_u(s)
\right)
ds
\right)
$$

其中指数逐 RGB 通道计算。

如果某条 ray 穿过更浑浊的水体区域，则路径积分更大：

$$
\int_0^t
\boldsymbol{\beta}
\left(
r_u(s)
\right)
ds
\uparrow
$$

对应透射率更小：

$$
\boldsymbol{T}_m(u,t)
\downarrow
$$

## 5. 理论渲染公式

每个 `scene Gaussian` 应该根据自己的有效深度 \(t_i(u)\) 受到不同的水体衰减，而不是所有 Gaussian 共用一个最终深度。

物体直接光项为：

$$
I_{\mathrm{obj}}(u)
=
\sum_i
T_i^{gs}(u)
\alpha_i(u)
c_i
\odot
\boldsymbol{T}_m(u,t_i)
$$

其中：

- \(t_i(u)\)：第 \(i\) 个 Gaussian 在像素 \(u\) 上的有效 ray depth。
- \(\odot\)：RGB 逐通道乘法。

简化的 backscatter 项先写成：

$$
I_{\mathrm{med}}(u)
=
b_\infty
\odot
\left(
1
-
\boldsymbol{T}_m(u,D_{\mathrm{vis}})
\right)
$$

其中：

- \(b_\infty\)：全局可学习水体散射颜色。
- \(D_{\mathrm{vis}}\)：当前 ray 的有效前景深度。

最终渲染为：

$$
I(u)
=
I_{\mathrm{obj}}(u)
+
I_{\mathrm{med}}(u)
$$

这套公式的含义是：

- 物体颜色先由 3DGS 负责。
- 每个 Gaussian 的物体光根据自己的路径积分衰减。
- 水体本身贡献额外的 backscatter / haze。
- 最终图像由物体项和水体项相加得到。

## 6. WaterSplatting 接口版本

第一版不建议直接改 CUDA kernel，让 kernel 在每个 segment 内重新查询 3D medium field。更稳的做法是：先把 3D medium field 沿每条 ray 积分成 WaterSplatting 现有接口需要的三张 per-pixel 图：

$$
\mathrm{medium\_attn}(u),
\quad
\mathrm{medium\_bs}(u),
\quad
\mathrm{medium\_rgb}(u)
$$

### 6.1 选择积分终点

对每个像素 \(u\)，选择参考积分长度：

$$
D_{\mathrm{ref}}(u)
=
D_{\mathrm{vis}}(u)
$$

第一版规则：

- 如果当前像素有可靠前景，优先使用渲染出的 `depth`。
- 如果当前像素没有可靠前景，使用 far plane 或场景 bbox 的 ray 交点。
- \(D_{\mathrm{ref}}(u)\) 必须和 renderer 内部 depth convention 保持一致。

### 6.2 生成 `medium_attn`

计算当前 ray 上的平均消光密度：

$$
\overline{\boldsymbol{\beta}}(u)
=
\frac{1}
{\max(D_{\mathrm{ref}}(u),\epsilon)}
\int_0^{D_{\mathrm{ref}}(u)}
\boldsymbol{\beta}
\left(
r_u(s)
\right)
ds
$$

然后作为 WaterSplatting 的 attenuation 输入：

$$
\mathrm{medium\_attn}(u)
=
\overline{\boldsymbol{\beta}}(u)
$$

这样 CUDA kernel 内部使用：

$$
\exp
\left(
-
\mathrm{medium\_attn}(u)
d_i
\right)
$$

来近似完整路径积分透射率：

$$
\exp
\left(
-
\int_0^{d_i}
\boldsymbol{\beta}
\left(
r_u(s)
\right)
ds
\right)
$$

这里 \(d_i\) 是 CUDA rasterizer 内部使用的 Gaussian depth。

### 6.3 生成 `medium_bs`

backscatter density 不建议完全自由学习，而是和 extinction density 绑定：

$$
\boldsymbol{\sigma}_b(x)
=
\boldsymbol{\rho}_b(x)
\odot
\boldsymbol{\beta}(x)
$$

其中：

$$
0
\leq
\boldsymbol{\rho}_b(x)
\leq
1
$$

实现时可以令：

$$
\boldsymbol{\rho}_b(x)
=
\mathrm{sigmoid}
\left(
g_\theta(x)
\right)
$$

这样天然保证：

$$
0
\leq
\boldsymbol{\sigma}_b(x)
\leq
\boldsymbol{\beta}(x)
$$

沿 ray 计算平均 scattering density：

$$
\overline{\boldsymbol{\sigma}}_b(u)
=
\frac{1}
{\max(D_{\mathrm{ref}}(u),\epsilon)}
\int_0^{D_{\mathrm{ref}}(u)}
\boldsymbol{\sigma}_b
\left(
r_u(s)
\right)
ds
$$

对应代码变量：

$$
\mathrm{medium\_bs}(u)
=
\overline{\boldsymbol{\sigma}}_b(u)
$$

### 6.4 生成 `medium_rgb`

第一版使用全局水体颜色：

$$
\mathrm{medium\_rgb}(u)
=
b_\infty
$$

其中：

$$
0
\leq
b_\infty
\leq
1
$$

实现时可以用一个可学习参数 \(\tilde b_\infty\)，再经过 sigmoid：

$$
b_\infty
=
\mathrm{sigmoid}
\left(
\tilde b_\infty
\right)
$$

如果后续想表达空间变化的水体颜色，再升级为：

$$
\mathrm{medium\_rgb}(u)
=
\overline{b}(u)
=
\frac{1}
{\max(D_{\mathrm{ref}}(u),\epsilon)}
\int_0^{D_{\mathrm{ref}}(u)}
b
\left(
r_u(s)
\right)
ds
$$

但第一版不建议这么做，因为它会增加可辨识性问题。

### 6.5 WaterSplatting 输出

WaterSplatting kernel 接收：

$$
\mathrm{medium\_attn},
\quad
\mathrm{medium\_bs},
\quad
\mathrm{medium\_rgb}
$$

输出：

$$
\mathrm{rgb\_object},
\quad
\mathrm{rgb\_medium},
\quad
\mathrm{pred\_image}
$$

最终关系为：

$$
\mathrm{pred\_image}
=
\mathrm{rgb\_object}
+
\mathrm{rgb\_medium}
$$

第一版实现时只需要保证 3D medium field 生成这三张 per-pixel medium 图，后面的 compositing 继续走原 WaterSplatting 主路径。

## 7. Loss 总体结构

总 loss 写成：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{medium}
\mathcal{L}_{medium}
+
\lambda_{scatter}
\mathcal{L}_{scatter}
+
\lambda_{decor}
\mathcal{L}_{decor}
$$

第一版推荐先实现：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{medium}
\mathcal{L}_{medium}
+
\lambda_{decor}
\mathcal{L}_{decor}
$$

如果 \(\boldsymbol{\sigma}_b\) 不是通过比例参数化自动满足合法范围，再加入：

$$
\lambda_{scatter}
\mathcal{L}_{scatter}
$$

## 8. RGB 重建 Loss

基础 RGB loss 负责让最终渲染图像接近 GT：

$$
\mathcal{L}_{rgb}
=
\left\|
I
-
I_{gt}
\right\|_1
+
\lambda_{ssim}
\left(
1
-
\mathrm{SSIM}
\left(
I,
I_{gt}
\right)
\right)
$$

其中：

- \(I\)：最终渲染图像，对应 `pred_image`。
- \(I_{gt}\)：GT 水下图像。
- \(\lambda_{ssim}\)：SSIM loss 权重。

这一项只约束最终图像像 GT，但它不能保证物体和水体分解合理。因此后面需要额外 loss。

## 9. Medium 平滑与低容量约束

medium field 应该表达低频水体分布，不应该表达物体纹理。对 3D medium field 加平滑和稀疏约束：

$$
\mathcal{L}_{medium}
=
\frac{1}
{|\mathcal{S}|}
\sum_{x\in\mathcal{S}}
\left\|
\nabla
\boldsymbol{\beta}(x)
\right\|_2^2
+
\lambda_\beta
\frac{1}
{|\mathcal{S}|}
\sum_{x\in\mathcal{S}}
\left\|
\boldsymbol{\beta}(x)
\right\|_1
$$

其中：

- \(\mathcal{S}\)：3D 采样点集合。
- \(\nabla \boldsymbol{\beta}(x)\)：medium field 在 3D 空间中的梯度。
- 第一项约束 medium 空间平滑。
- 第二项约束 medium 不要过强、不要到处解释颜色误差。

\(\mathcal{S}\) 的采样方式可以二选一：

- 从场景 bbox 内均匀采样 3D 点。
- 从训练 batch 的 camera rays 上采样 3D 点。

如果使用 `medium Gaussians`，还可以增加尺度约束：

$$
\mathcal{L}_{scale}
=
\frac{1}{K}
\sum_k
\max
\left(
0,
s_{\min}
-
s_k
\right)
$$

这个约束防止某个 `medium Gaussian` 变得过小，从而贴着物体边缘拟合纹理。

## 10. Scatter 合法性约束

物理上，backscatter density 不应该超过总消光密度。也就是：

$$
0
\leq
\boldsymbol{\sigma}_b(x)
\leq
\boldsymbol{\beta}(x)
$$

如果实现时没有使用比例参数化，可以加入 soft penalty：

$$
\mathcal{L}_{scatter}
=
\frac{1}
{|\mathcal{S}|}
\sum_{x\in\mathcal{S}}
\left\|
\max
\left(
0,
\boldsymbol{\sigma}_b(x)
-
\boldsymbol{\beta}(x)
\right)
\right\|_1
$$

但更推荐直接参数化为：

$$
\boldsymbol{\sigma}_b(x)
=
\mathrm{sigmoid}
\left(
g_\theta(x)
\right)
\odot
\boldsymbol{\beta}(x)
$$

这样 \(\mathcal{L}_{scatter}\) 可以不用显式计算。

## 11. Medium 不抢纹理的去相关 Loss

这是第一版最推荐的创新 loss。它的目的不是让最终图像更像 GT，而是防止 `medium field` 偷偷拟合物体纹理和边缘。

### 11.1 水体影响图

理论上定义水体影响图：

$$
M(u)
=
I(u)
-
J(u)
$$

其中：

- \(I(u)\)：最终水下渲染图像。
- \(J(u)\)：无水体时的物体渲染图像。
- \(M(u)\)：水体对图像造成的总影响。

如果实现里能输出无水颜色 \(J(u)\)，则用：

$$
M(u)
=
\mathrm{pred\_image}(u)
-
J(u)
$$

如果实现里暂时没有无水 \(J(u)\)，第一版直接使用：

$$
M(u)
=
\mathrm{rgb\_medium}(u)
$$

这是最稳的第一版，因为 `rgb_medium` 明确表示水体散射项。

### 11.2 物体纹理强度图

理想情况下，用无水颜色 \(J(u)\) 计算物体纹理强度：

$$
G_J(u)
=
\left\|
\nabla
J(u)
\right\|_1
$$

如果没有 \(J(u)\)，用 `rgb_object` 近似：

$$
G_J(u)
=
\left\|
\nabla
\mathrm{rgb\_object}(u)
\right\|_1
$$

### 11.3 水体高频强度图

对水体影响图 \(M(u)\) 计算梯度：

$$
G_M(u)
=
\left\|
\nabla
M(u)
\right\|_1
$$

对 RGB 图像 \(X(u)\)，实现时可以用有限差分：

$$
\left\|
\nabla X(u)
\right\|_1
=
\sum_{c\in\{r,g,b\}}
\left|
X_c(u+\Delta_x)
-
X_c(u)
\right|
+
\sum_{c\in\{r,g,b\}}
\left|
X_c(u+\Delta_y)
-
X_c(u)
\right|
$$

边界像素可以忽略，或者使用 padding 后再计算。

### 11.4 去相关约束

计算 \(G_M\) 和 \(G_J\) 的相关系数：

$$
\mathrm{Corr}(a,b)
=
\frac{
\sum_u
\left(
a(u)-\bar a
\right)
\left(
b(u)-\bar b
\right)
}{
\sqrt{
\sum_u
\left(
a(u)-\bar a
\right)^2
}
\sqrt{
\sum_u
\left(
b(u)-\bar b
\right)^2
}
+
\epsilon
}
$$

其中：

$$
\bar a
=
\frac{1}{N}
\sum_u
a(u)
$$

$$
\bar b
=
\frac{1}{N}
\sum_u
b(u)
$$

如果有有效 mask，则只在 mask 内计算 \(\bar a\)、\(\bar b\) 和相关系数。

去相关 loss 为：

$$
\mathcal{L}_{decor}
=
\left|
\mathrm{Corr}
\left(
G_M,
G_J
\right)
\right|
$$

这个 loss 的直觉是：

- 物体边缘和纹理会让 \(G_J\) 变大。
- 水体项 \(M\) 不应该在同一位置同步产生强边缘。
- 如果 \(G_M\) 和 \(G_J\) 高度相关，说明 `medium field` 正在复制物体纹理。
- 最小化 \(\mathcal{L}_{decor}\) 可以让 medium 更像低频水体，而不是自由纹理补丁。

第一版建议：

$$
M(u)
=
\mathrm{rgb\_medium}(u)
$$

$$
G_J(u)
=
\left\|
\nabla
\mathrm{rgb\_object}(u)
\right\|_1
$$

然后计算：

$$
\mathcal{L}_{decor}
=
\left|
\mathrm{Corr}
\left(
\left\|
\nabla
\mathrm{rgb\_medium}
\right\|_1,
\left\|
\nabla
\mathrm{rgb\_object}
\right\|_1
\right)
\right|
$$

## 12. 可选跨视角反水体一致性 Loss

这一项作为第二阶段或增强版，不建议第一版强制实现，因为它需要跨视角对应关系、depth、camera pose 和可见性判断。

如果同一个 3D 点在两个视角 \(v_1\)、\(v_2\) 中都可见，则反推去掉水体后的颜色：

$$
\hat J_v(u)
=
\frac{
I_{gt,v}(u)
-
I_{\mathrm{med},v}(u)
}{
\boldsymbol{T}_{m,v}
\left(
u,
D_v
\right)
+
\epsilon
}
$$

其中：

- \(v\)：视角编号。
- \(u\)：该 3D 点投影到视角 \(v\) 后的像素。
- \(D_v\)：该点在视角 \(v\) 下的 ray depth。
- \(I_{\mathrm{med},v}(u)\)：该视角下的水体散射项。
- \(\boldsymbol{T}_{m,v}(u,D_v)\)：该视角下从相机到该点的水体透射率。

跨视角一致性 loss：

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

这个 loss 的含义是：

- 同一个 3D 点从不同视角看到时，水体路径可以不同。
- 但去掉水体影响后，物体本身颜色应该一致。
- 这项 loss 直接约束新渲染公式里的路径积分 \(\boldsymbol{T}_m\) 和水体项 \(I_{\mathrm{med}}\)。

第一版可以先不实现 \(\mathcal{L}_{view}\)。等 `depth`、camera pose 和 visibility 稳定后，再把它作为增强项加入。

## 13. 第一版推荐实现顺序

第一版按下面顺序做：

1. 保持现有 WaterSplatting rasterizer 主路径不变。
2. 新增 3D medium field，用它沿 ray 生成 `medium_attn`、`medium_bs`、`medium_rgb`。
3. 渲染输出继续使用：

$$
\mathrm{pred\_image}
=
\mathrm{rgb\_object}
+
\mathrm{rgb\_medium}
$$

4. 第一版总 loss 使用：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{medium}
\mathcal{L}_{medium}
+
\lambda_{decor}
\mathcal{L}_{decor}
$$

5. 如果 \(\boldsymbol{\sigma}_b\) 没有通过 sigmoid 比例参数化，再加入：

$$
\lambda_{scatter}
\mathcal{L}_{scatter}
$$

6. 等训练稳定后，再加入跨视角：

$$
\lambda_{view}
\mathcal{L}_{view}
$$

## 14. 推荐权重设置

第一版可以从较弱正则开始：

$$
\lambda_{ssim}
=
0.2
$$

$$
\lambda_{medium}
\in
\left[
10^{-4},
10^{-2}
\right]
$$

$$
\lambda_{decor}
\in
\left[
10^{-3},
10^{-1}
\right]
$$

如果使用显式 \(\mathcal{L}_{scatter}\)：

$$
\lambda_{scatter}
\in
\left[
10^{-3},
10^{-1}
\right]
$$

调参原则：

- 如果 `rgb_medium` 明显复制物体边缘，提高 \(\lambda_{decor}\)。
- 如果 medium 到处变得很强，提高 \(\lambda_{medium}\) 里的稀疏项权重。
- 如果图像重建明显变差，先降低 \(\lambda_{decor}\) 和 \(\lambda_{medium}\)。
- 不要一开始就加入 \(\mathcal{L}_{view}\)，否则问题来源不好判断。

## 15. 验证方法

实现后至少验证下面几件事。

### 15.1 退化验证

关闭 medium 时：

$$
\mathrm{medium\_attn}
=
0
$$

$$
\mathrm{medium\_bs}
=
0
$$

$$
\mathrm{rgb\_medium}
=
0
$$

此时渲染应该退化为普通 3DGS / 物体项。

### 15.2 分解一致性

开启 medium 后，应满足：

$$
\mathrm{pred\_image}
=
\mathrm{rgb\_object}
+
\mathrm{rgb\_medium}
$$

可以直接计算：

$$
\left\|
\mathrm{pred\_image}
-
\mathrm{rgb\_object}
-
\mathrm{rgb\_medium}
\right\|_\infty
$$

该值应接近 0，只允许有浮点误差。

### 15.3 数值范围

需要检查：

$$
\mathrm{medium\_attn}(u)
\geq
0
$$

$$
\mathrm{medium\_bs}(u)
\geq
0
$$

$$
0
\leq
\mathrm{medium\_rgb}(u)
\leq
1
$$

如果使用比例参数化，还要检查：

$$
\mathrm{medium\_bs}(u)
\leq
\mathrm{medium\_attn}(u)
$$

### 15.4 可视化检查

训练过程中记录并可视化：

$$
\mathrm{medium\_attn},
\quad
\mathrm{medium\_bs},
\quad
\mathrm{medium\_rgb},
\quad
\mathrm{rgb\_object},
\quad
\mathrm{rgb\_medium},
\quad
\mathrm{pred\_image}
$$

期望现象：

- `rgb_object` 保留主要物体结构和纹理。
- `rgb_medium` 更平滑，主要表现为雾化、偏色和散射。
- `rgb_medium` 不应该明显复制物体边缘。
- `medium_attn` 和 `medium_bs` 不应该出现密集的高频纹理。

## 16. 最小实现版本总结

最小可实现版本只需要以下内容：

1. 一个低容量 3D medium field，输出：

$$
\boldsymbol{\beta}(x)
$$

2. 一个比例参数化的 backscatter density：

$$
\boldsymbol{\sigma}_b(x)
=
\mathrm{sigmoid}
\left(
g_\theta(x)
\right)
\odot
\boldsymbol{\beta}(x)
$$

3. 一个全局可学习水体颜色：

$$
b_\infty
=
\mathrm{sigmoid}
\left(
\tilde b_\infty
\right)
$$

4. 沿每条 ray 积分，生成：

$$
\mathrm{medium\_attn},
\quad
\mathrm{medium\_bs},
\quad
\mathrm{medium\_rgb}
$$

5. 继续使用 WaterSplatting kernel 得到：

$$
\mathrm{rgb\_object},
\quad
\mathrm{rgb\_medium},
\quad
\mathrm{pred\_image}
$$

6. 第一版 loss：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{medium}
\mathcal{L}_{medium}
+
\lambda_{decor}
\mathcal{L}_{decor}
$$

其中最有创新价值的是：

$$
\mathcal{L}_{decor}
=
\left|
\mathrm{Corr}
\left(
\left\|
\nabla
\mathrm{rgb\_medium}
\right\|_1,
\left\|
\nabla
\mathrm{rgb\_object}
\right\|_1
\right)
\right|
$$

它直接约束 `medium field` 不要拟合物体纹理，从而让物体-水体分解更符合水下成像逻辑。
