# 基于深度感知 Gaussian 特征融合的 3DGS 改进思路

## 核心想法

给每个 Gaussian 增加一个可学习特征，让它在渲染前融合邻域 Gaussian 的上下文信息，再由 MLP 预测参数残差，最后交给 gsplat 正常渲染。

```text
Gaussian 参数 + feature
        -> 局部邻域融合
        -> MLP 预测参数残差
        -> 更新 Gaussian 参数
        -> rasterization
        -> RGB / depth / alpha / loss
```

这个方向不是改 CUDA，也不是单纯加 loss，而是改 Gaussian 参数的生成 / 调整方式。

## 主要创新点

原版 3DGS 中，每个 Gaussian 基本独立优化：

```text
means, scales, rotations, opacity, color / SH
```

这里改成：

```text
每个 Gaussian 根据周围 Gaussian 的几何、颜色、深度信息修正自身参数。
```

水下场景中，颜色衰减和散射通常与深度强相关，因此可以设计深度感知的邻域融合模块。

一句话表述：

```text
提出一种基于深度感知的 Gaussian 邻域特征融合模块，用于建模水下场景中随深度变化的颜色衰减和散射退化。
```

## 推荐方案：3D kNN + Depth-Aware Fusion

不要做全局 self-attention。全局 attention 复杂度是：

```text
O(N^2)
```

3DGS 后期 Gaussian 数量很大，会很慢并且容易爆显存。

推荐做局部融合：

```text
1. 对每个 Gaussian 用 3D kNN 找 K 个邻居
2. 根据 3D 距离和 depth difference 计算权重
3. 加权融合邻居 feature
4. MLP 输出参数残差
5. 用更新后的参数渲染
```

## 融合权重

只考虑深度差异：

```text
w_ij = exp(- |depth_i - depth_j| / tau)
```

同时考虑 3D 距离和深度差异：

```text
w_ij = exp(- ||x_i - x_j||^2 / sigma^2) * exp(- |depth_i - depth_j| / tau)
```

其中：

```text
x_i, x_j: Gaussian 的 3D 位置
depth_i, depth_j: Gaussian 的相机深度
sigma: 控制空间距离影响
tau: 控制深度差异影响
```

融合特征：

```text
fused_i = sum_j w_ij * feature_j / sum_j w_ij
context_i = concat(feature_i, fused_i)
```

## 参数更新方式

建议预测残差，不要直接预测完整参数。

```text
delta_params_i = MLP(context_i)
```

可以先只修正颜色和不透明度：

```text
new_color_i = color_i + delta_color_i
new_opacity_i = opacity_i + delta_opacity_i
```

后续再尝试修正尺度：

```text
new_scale_i = scale_i * exp(delta_scale_i)
```

不建议一开始就改 `means` 和 `quats`，容易破坏几何稳定性。

## 配套 Loss：Gaussian-level Neighbor Consistency

除了像素级重建 loss，还可以加一个 Gaussian 级约束。

像素级 loss：

```text
L_photo = L1(rendered, gt) + lambda_ssim * (1 - SSIM(rendered, gt))
```

Gaussian 级 neighbor loss：

```text
L_neighbor = sum_ij w_ij * || refined_color_i - refined_color_j ||
```

其中 `i, j` 是两个 Gaussian，不是两个像素。

`refined_color_i` 表示第 i 个 Gaussian 经过邻域融合和 MLP 修正后的颜色：

```text
refined_color_i = color_i + delta_color_i
```

权重仍然使用深度感知邻域权重：

```text
w_ij = exp(- ||x_i - x_j||^2 / sigma^2) * exp(- |depth_i - depth_j| / tau)
```

含义：

```text
空间近、深度近的 Gaussian 应该有更一致的颜色修正。
空间远或深度差很大的 Gaussian 不强行一致。
```

总 loss 可以先写成：

```text
L_total = L_photo + lambda_neighbor * L_neighbor
```

第一版建议只对 `refined_color` 做一致性约束。后续再尝试 feature、opacity 或 depth consistency。

## 是否需要改 CUDA

不需要。

只要最终仍然输出 gsplat 需要的参数：

```text
means, quats, scales, opacities, colors
```

就可以直接调用：

```text
rasterize_fnc(...)
```

只有当要修改 Gaussian 内部 alpha compositing 或逐像素混合公式时，才需要改 CUDA。

## 和 SeaSplat / WaterSplatting 的关系

SeaSplat 更像：

```text
3DGS 渲染 RGB / depth / alpha
        -> 外部水下成像模型
        -> 水下 loss / 先验
```

WaterSplatting 更像：

```text
把水体衰减和散射写进 rasterizer 内部
```

当前这个想法属于：

```text
不改 rasterizer
改 Gaussian 参数调整模块
```

因此可以和 SeaSplat 的水下 loss 结合。

## 最小可行实验

建议第一版只做最小闭环：

```text
1. 每个 Gaussian 加 learnable feature
2. 3D kNN 找 K 个邻居
3. depth-aware weighted aggregation
4. MLP 输出 delta_color, delta_opacity
5. gsplat 正常渲染
6. 用 RGB loss / SSIM loss / 水下 loss 训练
```

推荐初始设置：

```text
feature_dim = 16
K = 8 or 16
只修正 color / opacity
不修正 means / quats
```

## 主要风险

```text
全局 attention 太慢，不建议做
feature 和 MLP 会增加训练时间
densification / pruning 时 feature 也要同步 clone / split / remove
参数残差过大可能破坏原始 3DGS 几何
如果没有水下深度退化动机，容易被认为只是堆模块
```

## 论文表述草稿

英文：

```text
We propose a depth-aware Gaussian neighborhood feature aggregation module to model depth-dependent color attenuation and scattering degradation in underwater scenes.
```

中文：

```text
我们提出一种深度感知的 Gaussian 邻域特征融合模块，使每个 Gaussian 在渲染前融合局部深度一致的邻域信息，从而更好地建模水下场景中的颜色衰减和散射退化。
```
