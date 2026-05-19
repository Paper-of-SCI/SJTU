# PUTrack 渐进式提示思路整理

来源：`3dgs/PUTrack_Improved_Underwater_Object_Tracking_via_Progressive_Prompting(1)(1).pdf`，并参考论文给出的官方仓库中 ARTrack_seq 的 UPT 实现命名；检查时对应官方仓库 commit 为 `b4ede83f9f82dda0f44f08ef352f86e407bb9a9f`。本文档目标是给后续 AI/开发者作为“写模块规格”使用，不是泛泛论文摘要。

## 1. 论文要解决的问题

已有水下目标跟踪大多走两条路：

1. `enhance-then-track`：先做水下图像增强，再把增强图送进普通开放场景 tracker。
2. `track-then-process`：先用普通 tracker 预测，再用后处理校正框。

这两类方案的问题是：增强、跟踪、后处理各自训练，目标不统一；水下数据少，容易过拟合；没有充分利用已经在大规模开放场景数据上训练好的 tracker 能力。

PUTrack 的核心思路是：不要重训整个 tracker，也不要外接图像增强/后处理，而是在 ViT tracker 的每个语义层旁边插入轻量的 `Underwater Prompter`，从当前 token flow 中挖出“水下场景提示”，再逐层传播并注入回主干。这样保留预训练 tracker 的通用跟踪能力，同时用少量参数学习水下域适配能力。

## 2. 一句话理解渐进式提示

渐进式提示不是一次性加一个 prompt token，也不是只在输入端加提示。它是在每个 ViT encoder layer 旁边都放一个 UPT：

```text
当前层 token -> UPT_l 生成当前水下 prompt
当前水下 prompt + 上一层 prompt -> prompt gate 融合
融合后的 prompt -> token gate 注入当前 token
注入后的 token -> 原 ViT encoder layer
融合后的 prompt -> 传给下一层 UPT
```

所以 prompt 会沿着网络深度逐层更新、逐层注入。低层更偏局部纹理和退化细节，高层更偏语义区分和前后景关系；逐层传播可以让水下提示覆盖所有语义层，而不是只影响某一层。

## 3. 基础 tracker 抽象

论文把 ViT-based 开放场景 tracker（论文称 `open-air tracker`）抽象成：

```text
template image Z
search region X
    -> patch embedding + position embedding
    -> template tokens H_Z, search tokens H_X
    -> concat 得到 H_0
    -> N 层 ViT encoder
    -> prediction head / decoder
    -> bbox B
```

公式化写法：

```text
H_Z, H_X = E_embedding(Z_p, X_p) + E_pos
H_0 = Concat(H_Z, H_X)
H_l = E_l(H_{l-1}), l = 1..N
B = phi(H_N)
```

PUTrack 不改变 tracker 的 prediction head 或 decoder。OSTrack 可以继续用 corner prediction head，ARTrack 可以继续用 autoregressive decoder。改动集中在 ViT encoder 的侧边 prompt 分支。

## 4. 总体架构

PUTrack 的模块由四部分组成：

1. 冻结的基础 tracker：保留预训练权重，主要负责原有的通用跟踪能力。
2. 每层一个 UPT：从当前层 token 中提取水下 prompt。
3. 层间 prompt gate：融合当前层 prompt 和上一层 prompt，形成可逐层传递的 prompt flow。
4. token 注入 gate：控制 prompt 以多大强度加回 token flow。

训练时只更新 prompt 相关参数，基础 tracker 绝大多数参数冻结。论文在 ARTrack 上只引入约 `0.6M` 可训练参数，约占原模型 `0.4%`。

## 5. UPT 要学什么

论文认为水下场景对开放场景 tracker 的主要困难有三类：

1. 水下生物群体出现，外观相似，容易被 similar distractor 干扰。
2. 生物拟态导致前景和背景边界不清，目标会融入珊瑚、岩石、水草等背景。
3. 水下成像退化，包括模糊、暗光、颜色偏移、浑浊，细节信息弱。

ViT 擅长全局依赖和长程关系，但局部高频细节捕获弱。UPT 因此采用轻量卷积分支，从 token flow 中提取局部细节、高频信息和局部稳健特征，作为水下 prompt 注入 ViT。

## 6. UPT 内部结构

给定当前层 token flow `H_l`，先把 token reshape 成特征图：

```text
H_l: [B, L, C]
F_l: [B, C, H, W]
```

注意：实现时只有当 `L = H * W` 且能恢复二维空间布局时才能直接 reshape。官方 ARTrack_seq 实现里，UPT 主要作用在 search tokens 上，因为 search token 数为 `16 * 16 = 256`，能变成二维特征图；template tokens 后面再拼回主 token flow。

UPT 三步：

### 6.1 降维投影

先用 `1x1 conv` 降到低维空间，减少冗余：

```text
M = f1(F_l)
f1: Conv2d(C, C / beta, kernel_size=1)
beta = 24
```

若 `C = 768`，则 `C / beta = 32`。官方实现里对应 `conv_in: 768 -> 32`，再分成两个 `16` 通道分支。

### 6.2 双分支提取

把低维特征按通道切成两支：

```text
M1, M2 = split(M, dim=channel)
M1: [B, C/(2*beta), H, W]
M2: [B, C/(2*beta), H, W]
```

分支 A：卷积分支，强调局部纹理和高频细节。

```text
M3 = Conv3x3(Linear_or_Conv1x1(M1))
```

官方代码是：

```text
conv1: 1x1 conv
proj1: 3x3 conv
GELU
```

分支 B：池化分支，强调局部区域的稳健聚合。

```text
M4 = Linear_or_Conv1x1(MaxPool(M2))
```

官方代码是：

```text
MaxPool2d(kernel=3, stride=1, padding=1)
proj2: 1x1 conv
GELU
```

### 6.3 融合并升维回 token 维度

两支 concat 后用 `1x1 conv` 融合，再投影回原通道数：

```text
P_cur_l = f3(Concat(M3, M4))
f3: Conv2d(C / beta, C, kernel_size=1)
P_cur_l: [B, C, H, W] -> [B, L, C]
```

官方代码中是：

```text
conv_fuse: 1x1 conv, 32 -> 32
proj:      1x1 conv, 32 -> 768
```

这里的 `P_cur_l` 不是独立 token，而是与原 token 空间位置对齐的 dense prompt。它可以直接加回 token flow。

## 7. 层间 prompt gate

UPT 每层都会产生一个当前层 prompt `P_cur_l`。为了让 prompt 逐层传播，PUTrack 不直接丢掉上一层 prompt，而是把上一层 prompt `P_{l-1}` 和当前层 prompt 做门控融合。

更适合按官方实现理解为：

```text
g_l = sigmoid(gamma_l)
P_l = (1 - g_l) * P_cur_l + g_l * P_{l-1}
```

初始化：

```text
gamma_l = -10
sigmoid(-10) ≈ 0
```

因此训练初期 `P_l` 基本等于当前层 UPT 输出，上一层 prompt 只占很小比例。随着训练，gate 可以学习是否更多保留上一层 prompt。

论文 PDF 中 prompt gate 的公式、初始化解释和官方实现之间有一个容易混淆的权重方向问题：如果只机械照公式，很容易得到“初始偏上一层 prompt”的行为；但论文文字意图和官方代码都更接近“初始以当前层 prompt 为主，只少量继承上一层”。实现时建议按官方代码的 `Gate_Prompt` 方向写：输入 `xin` 是上一层 prompt，`xout` 是当前层 prompt，输出为：

```python
gate = sigmoid(gate_logit)      # gate_logit init -10
out = (1 - gate) * current + gate * previous
```

这个 gate 的意义是：让 prompt flow 不是每层完全独立，而是在“当前水下线索”和“历史层水下线索”之间自适应平衡。

## 8. token 注入 gate

得到融合 prompt `P_l` 后，还要控制它注入 token flow 的强度：

```text
H_prompted_l = H_l + t_l * P_l
t_l = sigmoid(delta_l)
```

论文中 `t_l` 是按 token 位置的门控权重，可理解成 `[1, L]` 或广播到 `[B, L, C]`。初始化：

```text
delta_l = 10
sigmoid(10) ≈ 1
```

因此训练初期大部分 prompt 都会被加进 token flow。官方代码中对应 `Gate_Feature`：

```python
gate_logit init +10
out = token_in + sigmoid(gate_logit) * prompt
```

这个 gate 的意义是：不是所有空间位置都需要同样强的水下提示。相似干扰物、暗光区域、前后景边界附近可能需要更强 prompt；容易跟踪的区域可以弱一些。

## 9. 每层前向流程

抽象实现如下：

```python
prompt_prev = None

for l in range(num_layers):
    token_for_prompt = select_prompt_tokens(tokens)
    token_norm = prompt_norm[l](token_for_prompt)

    # [B, L, C] -> [B, C, H, W] -> UPT -> [B, L, C]
    prompt_cur = feature2token(UPT_l(token2feature(token_norm)))

    if prompt_prev is None:
        prompt = prompt_cur
    else:
        prompt = prompt_gate[l - 1](previous=prompt_prev, current=prompt_cur)

    token_for_prompt = token_gate[l](token_for_prompt, prompt)
    tokens = merge_back(tokens, token_for_prompt)

    tokens = encoder_layer_l(tokens)
    prompt_prev = prompt
```

如果复现官方 ARTrack_seq 版本，则更接近：

```python
# patch embedding 后先对 search token 做第 0 层 prompt
x_search = patch_embed(search)
z_template = patch_embed(template)

p = UPT_0(token2feature(norm_0(x_search)))
x_search = x_search + sigmoid(delta_0) * p

# 加 position embedding / identity embedding 后拼接
tokens = combine_tokens(z_template, x_search)

for i, block in enumerate(vit_blocks):
    if i >= 1:
        z_tokens = tokens[:, :lens_z]
        x_search = tokens[:, lens_z:]

        p_cur = UPT_i(token2feature(norm_i(x_search)))
        p = prompt_gate_i(previous=p, current=p_cur)
        x_search = x_search + sigmoid(delta_i) * p

        tokens = combine_tokens(z_tokens, x_search)

    tokens = block(tokens)
```

核心点：UPT 是旁路模块，不替代 ViT block；prompt 是 residual additive signal，不改变原 tracker 的 head/decoder 接口。

## 10. 为什么叫 progressive

这里的 progressive 有三层含义：

1. 层级渐进：每个 encoder layer 都有自己的 UPT，prompt 从浅层传到深层。
2. 语义渐进：浅层 prompt 更偏局部纹理、边缘、暗光、颜色退化；深层 prompt 更偏目标身份、前后景区分、相似目标抑制。
3. 训练渐进：基础 tracker 冻结，只让轻量 prompt 参数学习水下域差异，避免小规模水下数据把通用 tracker 训练坏。

它和普通 prompt token 的区别：

```text
普通 prompt token：通常是若干可学习 token，拼到序列里，通过 attention 间接影响主 token。
PUTrack prompt：由当前图像 token 动态生成，与空间位置对齐，直接 residual 注入 token flow，并跨层传播。
```

论文消融显示，额外拼接 trainable prompt tokens 没有带来明显提升，甚至会干扰 attention；PUTrack 的 dense prompt + gate 更适合水下跟踪。

## 11. 训练方式

训练策略：

1. 加载预训练开放场景 tracker。
2. 冻结基础 tracker 主体参数。
3. 只训练 prompt 相关参数：UPT、prompt norm、prompt gate、token gate。
4. 使用水下训练集，同时混入开放场景训练集，避免过拟合小规模水下数据。

论文设置：

```text
训练集混合：LaSOT-train : GOT-10K-train : HUT290-train = 1 : 1 : 1
PU-OSTrack：prompt module 训练 15 epochs，第 10 epoch 后学习率降到 1/10
PU-ARTrack：prompt module 训练 20 epochs，CE loss weight = 2，学习率 = 8e-5
```

训练收益：

```text
只新增约 0.6M trainable parameters
显存峰值比 full fine-tuning 低约 34%
训练 epoch 从 full fine-tuning 的 75 降到 prompt tuning 的 20
```

## 12. 论文消融对实现的约束

论文的消融结果对后续实现有几个直接约束：

1. 不要优先做 full fine-tuning。完整微调整个 ARTrack 虽然能提升水下性能，但 PUTrack 只训练少量 prompt 参数，在 UTB180、HUT290 上比 full fine-tuning 更好，并且训练成本更低。
2. 不要简单叠加 Adapter 或普通 prompt token。论文尝试把 UPT 和 Adapter、trainable prompt tokens 结合，结果没有稳定提升；prompt token 还可能干扰 attention，导致性能下降。
3. 两个 gate 都要保留。消融表明 prompt gate 和 token gate 的组合能带来稳定收益，尤其在 UTB180 上更明显。实现时不要为了简化直接把 prompt 无门控相加。
4. `beta = 24` 是默认低维投影设置。UOT100 上 `beta = 16` 略好，但 UTB180 和 HUT290 上 `beta = 24` 最优或更稳；论文最终选 `24` 是性能和效率的折中。
5. prompt 应该改善 attention，而不是只改数值特征。论文可视化显示，加入 PUTrack 后，encoder attention 更集中在目标上，对相似干扰物、拟态背景和遮挡场景更稳。
6. 轻量化是核心目标。论文报告新增约 `1.8G` FLOPs 和 `0.6M` 参数，推理速度相比 baseline 有小幅下降，但训练峰值显存降低约 `34%`，训练轮数也明显减少。

所以后续写模块时，优先保证“动态 UPT + 层间 gate + token gate + 冻结主干”这条主线，而不是把它扩展成复杂适配器堆叠。

## 13. 实现模块建议

建议拆成独立、可拆卸模块：

```text
UnderwaterPrompter
  - 输入: feature map [B, C, H, W]
  - 输出: prompt feature [B, C, H, W]
  - 内部: 1x1 降维 -> 双分支 -> concat -> 1x1 升维

PromptGate
  - 输入: previous_prompt, current_prompt
  - 输出: fused_prompt
  - 参数: gate_logit，标量或按 token/channel 扩展
  - 初始化: -10

TokenPromptGate
  - 输入: token_flow, fused_prompt
  - 输出: prompted_token_flow
  - 参数: gate_logit，建议按 token 位置
  - 初始化: +10

ProgressivePromptAdapter
  - 持有每层 UPT / norm / gate
  - 负责在每个 ViT block 前生成、融合、注入 prompt
  - 不负责 bbox head，不改 tracker 输出协议
```

接口上不要把 PUTrack 写死到某个 tracker。更稳的接口是：

```python
class ProgressivePromptAdapter(nn.Module):
    def forward_layer(
        self,
        layer_idx: int,
        tokens: Tensor,
        prompt_state: Tensor | None,
        layout: TokenLayout,
    ) -> tuple[Tensor, Tensor]:
        ...
```

`TokenLayout` 至少要能描述：

```text
哪些 token 可用于 UPT
这些 token 如何 reshape 成 H x W
prompt 注入后如何 merge 回原 token flow
template/search token 的边界在哪里
```

这样模块可以迁移到 OSTrack、ARTrack 或其它 ViT backbone。

## 14. 关键注意点

1. 不要把 UPT 做成图像增强模块。它处理的是 token/feature，不直接改 RGB 图。
2. 不要把 prompt 做成固定参数表。PUTrack 的 prompt 是从当前样本的 token 动态生成的。
3. 不要默认 concat 后的 template+search tokens 可以直接 reshape。很多 tracker 中 `L_z + L_x` 不是平方数，官方实现主要对 search tokens 做 `token2feature`。
4. 不要训练整个 backbone。PUTrack 的价值在于冻结基础 tracker，只训练 prompt 侧边模块。
5. 不要改 prediction head 的输入语义。注入 prompt 后的 token 仍应保持原 tracker 期望的 shape 和 token 顺序。
6. gate 初始化很重要：prompt gate 初始偏当前层 prompt，token gate 初始强注入 prompt。
7. 若要做严格论文复现，prompt gate 方向建议以官方代码为准；论文 PDF 的公式和初始化解释存在容易混淆的地方。

## 15. 可迁移到其它模块的抽象

如果后续不是做 2D 跟踪，而是把这个想法迁移到其它水下视觉模块，可以保留以下本质：

```text
冻结一个已有强 backbone
在每个语义层旁边加可训练 side prompter
prompter 从当前层表示中动态挖出水下域提示
提示跨层门控传播
提示以 residual/gated 方式注入主干表示
最终 head 尽量不变
```

迁移时需要重新定义三件事：

1. `token flow` 是什么：图像 patch token、Gaussian token、点云 token，还是多视角 feature token。
2. `token2feature` 怎么做：是否有规则二维布局；没有二维布局时不能直接用 2D conv，可换成 MLP、1D conv、graph/local-neighbor op。
3. `prompt 注入位置` 在哪里：每层 transformer 前、attention 前、FFN 前，或某个局部更新模块前。

只要保持“旁路生成、跨层传播、门控注入、主干冻结”这四点，就保留了 PUTrack 渐进式提示的核心。
