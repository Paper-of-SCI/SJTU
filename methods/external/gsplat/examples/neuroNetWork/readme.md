# GAE输入维度

## means

**不是只把原始坐标 `x, y, z` 喂给 MLP，而是把每个坐标变成一组不同频率的 sin/cos 特征。**

假设一个 GS 的位置是`xyz = [x, y, z]`

原始输入只有 3 个数：`x, y, z`

做 sinusoidal positional encoding 后，会变成：

```
x, y, z,

sin(1*pi*x), sin(1*pi*y), sin(1*pi*z),
cos(1*pi*x), cos(1*pi*y), cos(1*pi*z),

sin(2*pi*x), sin(2*pi*y), sin(2*pi*z),
cos(2*pi*x), cos(2*pi*y), cos(2*pi*z),

sin(4*pi*x), sin(4*pi*y), sin(4*pi*z),
cos(4*pi*x), cos(4*pi*y), cos(4*pi*z),

sin(8*pi*x), sin(8*pi*y), sin(8*pi*z),
cos(8*pi*x), cos(8*pi*y), cos(8*pi*z)
```

> 因为 MLP 对原始坐标 `[x, y, z]` 的高频空间变化不敏感。加上 sin/cos 后，位置稍微变化，某些高频特征就会明显变化，MLP 更容易学到“这个 GS 在空间哪里、附近结构有什么细节”。

`原始 xyz`: $xyz \in \mathbb{R}^{N\times 3}$

`sin 特征: 4个频率 * 3维` = 12维

`cos 特征: 4个频率 * 3维` = 12维

`总共`: $\mathbf{p}_{enc} \in \mathbb{R}^{N \times 27}$

---

## opacities

`总共`: $\alpha = \sigma(o), \quad \alpha \in \mathbb{R}^{N \times 1}$

---

## quats

`总共`: $\hat{\mathbf{q}} = \mathrm{normalize}(\mathbf{q}), \quad \hat{\mathbf{q}} \in \mathbb{R}^{N \times 4}$

---

## scales

`总共`: $\mathbf{s} \in \mathbb{R}^{N \times 3}$

## color

3阶SH系数
`总共`: $\mathbf{c}_{SH} \in \mathbb{R}^{N \times 16 \times 3} = \mathbf{c}_{SH} \in \mathbb{R}^{N \times 48}$

## 总维度

27+1+4+3+48=83

# 用法示例

```python
import torch
from MLP.gaussian_attribute_encoder import GaussianAttributeEncoder

device = "cuda"

encoder = GaussianAttributeEncoder(
    input_dim=83,
    feature_dim=32,
    hidden_dim=64,
    num_layers=3,
    normalize_output=True,
).to(device)

num_GS = 100000
attributes = torch.randn(num_GS, 83, device=device)

features = encoder(attributes)

print(features.shape)
# torch.Size([100000, 32])
```
