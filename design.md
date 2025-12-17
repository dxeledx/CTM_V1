### 模块链路（建议 v0 先最小闭环）
1. **Epoch / 标准化（Trial-wise）**
    - 输入：`X ∈ R^{C×T}`（22×1000）
    - 操作：trial 内 Z-score（或后面再换 EMS/robust scaler）
    - 说明：很多强 baseline 也会用“简单标准化 + 强架构/增强”路线 
2. **（可选）对齐层 EA / 不对齐（先留接口）**
    - EA 典型做法：对每个 domain 计算平均协方差作为参考矩阵，再做白化对齐 
    - **注意一个关键坑：EA 会导致“对齐后通道不再对应固定电极位置”**（通道变成混合权重），这会影响你后续“电极位置 embedding / spatial token”的设计
    - 所以：**tokenizer 若强依赖通道拓扑**，EA 的位置要非常谨慎（或改用不依赖通道语义的一类 tokenizer）。
3. **增强层（train only）**
    - **S&R（Segmentation & Recombination）**：把 trial 切成 K 段，从同类不同 trial 抽段按时间拼回去，保序不破坏时序结构 
    - 这是很多 MI 深度模型常用增强之一，且写进论文方法里非常好解释。
4. **EEG Tokenizer（核心！tokens 从何而来）** → 产出 `KV tokens`
    - 输入：`X ∈ R^{B×C×T}`
    - 输出：`E ∈ R^{B×N×d_model}`（N 个 token，每个 d_model 维）
    - **v0 推荐：Conv-Patch Tokenizer（CTNet/Conformer 这类范式）**
        * Temporal Conv：沿时间提频带/节律特征
        * Spatial/Depthwise Conv：跨通道混合空间模式
        * Pooling/Stride：把时间压成 token 序列（N 就是 token 数）
        * 重要：**token size/池化尺度会显著影响性能**，过大过小都不行
5. **CTM Core（“思考引擎”）**：多 internal ticks 迭代读 tokens
    - CTM 有一个**与数据维度解耦的 internal tick 轴**，能对静态输入迭代 refine
    - 每个 tick：
        * 用历史活动计算**synchronization 表示**`S_t = Z_t Z_t^T`
        * 从 synchronization 投影出 query，对 tokens 做 cross-attention（tokens 是 keys/values）
        * synapse + NLM 更新神经元状态（产生更丰富 dynamics）
        * 输出：每个 tick 一份 logits（或同步表示再接 head）
6. **Head / 聚合策略（把多 tick 变成一个预测）**
    - v0 最贴 CTM：用“最小 loss tick + 最大 certainty tick”的组合损失来训练，使其具备“原生自适应计算”倾向
    - 推理时：可取 `t = argmax(certainty)` 的那一步 logits，或对后若干步做平均（后续再做消融）。

下面先把 **BCI-IV-2a 在 LOSO 场景下，“Epoch 截取 + Trial-wise 标准化”** 这一段写成接近论文 _Methods_ 可直接复用的 protocol，并对齐近几年常见（Q1/Top）做法，顺带把坑点标出来。

---

## 1) Epoch（trial）截取：用哪一个事件当对齐点？
**数据集事实**：BCI-IV-2a 每个 trial 在 _t=0s_ 出现注视十字，_t=2s_ 出现方向 cue（对应类别），_t=6s_ 十字消失；事件码里 **768=trial start，769–772=cue onset（四类），1023=reject trial**。 

### 推荐对齐策略（最常用、最“干净”）
+ **对齐点**：用 cue onset（769–772）作为 epoch 的零点（t=0）。
+ **时间窗**：截取 **[0, 4] s**（即 cue 后 0–4 秒），这等价于论文里常写的 **[2,6] s of each trial**。
    - 这就是 Conformer 在 IV-2a 上的标准写法：用 **[2,6] 秒**并做 [4,40]Hz 带通。

这样做的好处：窗口几乎只包含运动想象主段（ERD/ERS 更集中），也更不容易被审稿人质疑“用到了 cue 前的视觉/准备期成分”。

### 可作为消融的备选窗（有论文这么做）
+ **[-0.5, 4] s**（相当于 [1.5,6] s）：把 cue 前 0.5 秒“准备期/视觉诱发残留”也纳入，有时对泛化有帮助，但更容易被质疑“不是纯 MI”。例如 EEG-DBNet 在 IV-2a 上采用 **[1.5,6] s**。

结论：主实验建议 **[0,4] s @ cue-onset 对齐**；把 **[-0.5,4] s** 留作 ablation（你要冲一区，最好把“窗选择影响”做成一张表或一张曲线）。

---

## 2) Trial-wise 标准化：怎么做才既强又“绝不 transductive”？
你要避免审稿人卡“transductive/泄露”，核心规则是：**测试被试的数据不能参与任何统计量拟合**。Trial-wise 的优势在于：**每个 trial 只用自己的统计量**，天然不会泄露跨 trial / 跨被试信息。

### 文献里两类常见做法（对比）
**A. 训练集统计量标准化（不是 trial-wise）**  
Conformer 明确写：Z-score 的均值/方差用训练数据计算，然后直接用到测试数据。 

+ 在 LOSO 里如果你严格“只用训练被试”算，这也不算泄露；
+ 但它**不是**你想要的“trial-wise”，而且对跨被试幅值漂移更敏感。

**B. 真·Trial-wise Z-score（每个 trial 一套 μ/σ）**  
CTNet 明确采用 **对每个原始 trial (X_i) 计算均值 μ 和标准差 δ 来做 Z-score**。 

+ 这是最容易写进 Methods 且最不怕审稿人挑刺的方案之一。

---

## 3) 我建议你采用的 Trial-wise 标准化（更贴 MI 的“数学折中”，也方便作为创新点）
MI 的判别信息很大一部分体现在**通道间相对能量差异**（如 C3/C4 附近 ERD），所以要小心：**“每通道都缩放到单位方差”可能会把能量差异抹平**。

因此我建议的默认版本是一个折中（你可以把它当作你们 pipeline 的“第一处可写成数学创新”的点）：

### 推荐默认：**per-channel 去均值 + 全局（跨通道）缩放**
对每个 epoch (X\in\mathbb{R}^{C\times T})（C=22, T=1000）：

1. **每通道去直流/偏置：**  
[  
\mu_c=\frac{1}{T}\sum_{t=1}^{T}X_{c,t},\quad \tilde X_{c,t}=X_{c,t}-\mu_c  
]
2. **用整个 trial 的全局尺度做缩放（保留通道间相对能量）：**  
[  
\sigma=\sqrt{\frac{1}{CT}\sum_{c=1}^{C}\sum_{t=1}^{T}\tilde X_{c,t}^2},\quad  
X'_{c,t}=\frac{\tilde X_{c,t}}{\sigma+\varepsilon}  
]
+ 这仍然是 **trial-wise**（每个 trial 自己算 (\mu_c,\sigma)），**绝无 transductive 嫌疑**；
+ 相比 CTNet 的“直接用整个 trial 的 μ/δ 做 Z-score”，它更强调“先保证每个通道零均值”，同时缩放不破坏通道相对强弱。

### 可作为补强/稳健版本（抗伪迹）
如果你担心个别 trial 有爆点伪迹，把均值/方差换成 robust 统计量：

+ (\mu_c \leftarrow \text{median}_t(X_{c,t}))
+ (\sigma \leftarrow 1.4826 \cdot \text{median}(|\tilde X_{c,t}|))

这个也完全 trial-wise，而且对偶发大幅伪迹更稳。

---

下面把我们项目里“增强层（train only）”这一步，写成可以直接落地实现的**骨架 + 细化选项**。

---

## 增强层骨架（Train only）：S&R 模块
### 输入/输出接口
+ 输入：标准化后的 epoch (X \in \mathbb{R}^{C\times T})，标签 (y\in{1,\dots,4})
+ 输出：增强后的 (\tilde X)（同形状），标签不变 (\tilde y=y)
+ **只在训练阶段启用**；val/test 完全关闭（LOSO 下尤其重要）

### Baseline S&R（复刻顶刊常用版本）
对同一类别 (c) 的训练样本集合 (\mathcal{D}_c={X_i}_{i=1}^{M})，把每个 (X_i) 沿时间等分成 (K) 段：  
[  
X_i = [X_i^{(1)}, X_i^{(2)}, \dots, X_i^{(K)}],\quad X_i^{(k)}\in\mathbb{R}^{C\times (T/K)}  
]  
生成一个新样本：  
[  
\tilde X = [X_{r_1}^{(1)}, X_{r_2}^{(2)}, \dots, X_{r_K}^{(K)}],\quad r_k \sim \text{Unif}{1,\dots,M}  
]  
并且**按原时间顺序拼接**（只换“段的来源 trial”，不打乱段的时序）。这就是 Conformer/CTNet 描述的核心机制。

### BCI-IV-2a 的默认超参建议（可作为主设置）
+ 你用的 epoch 长度是 cue 后 4 秒，250 Hz ⇒ (T=1000)
+ 建议先设 **(K=8)**（每段 125 点≈0.5s），这也是 Conformer 文中给出的常用选择（他们实验里就设了 (N_s=8)）。
+ 训练时每个 iteration：
    - 生成 **与原 batch 同样数量**的增强样本（Conformer 就这么做）。
    - 训练用 “原样本 + 增强样本” 拼成 2B（或者用概率 (p) 替换一部分为增强版，保持 batch size 不变）

---

## LOSO 场景下“不会踩坑”的实现细则
### 1) 绝对禁止的信息流
+ **测试被试**的数据不能进入任何形式的“增强供体池”（包括无标签也不行，除非你明确写 transductive setting）
+ val 被试也同理：用于 early stopping 的那名被试，不能被拿来当 S&R 供体

### 2) batch 内同类样本不足怎么办？
S&R 需要“同类供体”。真实训练时某个 batch 里某类可能只有 1 条，拼不起来。

建议做一个 **class-wise memory bank（仅训练被试）**：

+ 每类维护一个队列，存最近若干个该类 epoch（或它们的 segment）
+ 采样 (r_k) 时从 memory bank 取，而不是强依赖当前 batch

这点工程上很关键，不然你的增强会在训练早期“偶尔可用、偶尔失效”。

### 3) 拼接边界的“谱泄露/不连续”问题
S&R 虽然保序，但不同 trial 的段在边界处会有幅值/相位不连续，可能引入高频伪迹。

一个很实用且可写成“方法小创新”的修补：**边界 cross-fade（余弦窗平滑）**  
设段与段交界处用长度 (L_b) 的过渡窗（比如 10–25 个采样点）：  
[  
\tilde X[:,t] \leftarrow w(t),A[:,t] + (1-w(t)),B[:,t],\quad w(t)=\tfrac12\left(1+\cos\frac{\pi t}{L_b}\right)  
]  
这样能显著减少拼接导致的高频尖峰，同时不改变宏观时序结构（审稿也容易理解：提升“物理合理性”）。

---

## 你想要的“可写进论文的增强创新”（可选，不强求）
在 baseline S&R 上，我们可以加一个**相似性引导的供体选择**（保证拼出来的 trial 更“像真数据”）：

### Similarity-Guided S&R（SG-SR）
对每个 segment 位置 (k)，不再均匀随机选供体 trial，而是按“段级特征相似度”加权采样：

+ 先定义一个段特征 (g(X^{(k)}))（比如该段的各通道对数方差/带功率向量）
+ 对候选供体 (j) 计算距离 (d_{ij}^{(k)}=|g(X_i^{(k)})-g(X_j^{(k)})|_2^2)
+ 采样概率：  
[  
P(r_k=j) \propto \exp\Big(-\frac{d_{ij}^{(k)}}{\tau}\Big)  
]  
这样拼出来的相邻段在统计上更一致，减少“拼接伪迹”，尤其利于跨被试泛化（因为它逼迫模型学更稳定的结构，而不是记住某些异常边界）。

下面我们把 **EEG Tokenizer（把原始EEG变成可供CTM cross-attn读取的tokens）** 这一步先“定骨架 + 定接口 + 定可消融开关”。你说得很对：**token从何而来**基本决定了后面CTM/Transformer到底在学什么。

---

## 1) 先对齐：CTM需要的“token接口”是什么？
CTM本质是在每个 internal tick 用 **由同步表征投影出来的 query** 去 cross-attention 你的“数据特征tokens（K,V）”，得到 `o_t` 再回灌到内部动力学里。论文里写得很清楚：`o_t = Attention(Q=q_t, KV=FeatureExtractor(data))`，并且 `o_t` 会和内部状态一起进入下一步计算。  
伪代码里也明确是：`kv = kv_projector(backbone(inputs))`，然后 `attn(q, kv, kv)`。

**所以我们Tokenizer要产出：**

+ `tokens`：形状建议 **[B, N, d_kv]**（N别太大，不然CTM每个tick都attend会爆算力）
+ （可选）`mask/pos`：时序位置编码 or 电极拓扑位置编码

---

## 2) 近5年高水平论文里，“EEG→tokens”最常见的三种范式
### 范式A：Shallow Conv Encoder → Patch/Time tokens（最主流、最稳）
典型做法就是“先用CNN把EEG变成浅层时空特征，再用pool/stride把时间切成tokens”。

+ **EEG-Deformer（J-BHI方向的工作）**：强调“**100ms微状态**”启发，把 temporal conv kernel 设为 `(1, 0.1*fs)`，再用 `(c,1)` 做空间卷积，maxpool后 **rearrange成tokens + learnable position encoding**。
+ **CTNet**：明确指出第二个pool核 `P2` 决定 token size（序列长度），并讨论 **token过大=过平滑丢细节，过小=易受局部噪声**，并在IV2a上扫到一个最优token size区间。
+ **Conformer类**（卷积+Transformer）也同样是“卷积嵌入→pool形成patch tokens→Transformer”。 

这条路线的优点：贴合EEG低信噪比与时频局部性；也最容易做严谨的LOSO且不搞“跨被试统计量泄露”。

### 范式B：Filter-bank / multi-view → tokens（更强的MI先验）
MI强依赖 μ/β节律，很多方法把“多频带视图”显式塞进网络。比如 FBCNet强调 **multi-view filter-bank + variance层聚合时域信息**。   
这给我们的启发是：Tokenizer里可以做 **多分支时间卷积(不同kernel≈不同频带/时间尺度)**，让token天生带“频带语义”。

### 范式C：Channel tokens / Topography-aware tokens（更“结构化”，但跨数据集麻烦）
把“电极当token”，再加2D位置编码/图结构。对BCI-IV-2a（电极固定）很香，但你后面如果真要“喂所有公开数据集”，不同montage会让这条路线工程复杂度上升（需要montage对齐/插值/learnable mapping）。

---

## 3) 给我们CTM配的“Tokenizer骨架”（推荐v1：范式A + 可选B）
### 输入
+ 原始epoch：`X ∈ R^{B×C×T}`（BCI-IV-2a：C=22, T=1000 对应2–6s@250Hz）

### Tokenizer v1（强烈建议这样起步）
**(1) Multi-scale Temporal Conv（频带/时间尺度先验）**  
做 K 个分支，每个分支一个 temporal conv（只沿时间卷）：

+ kernel_ms 可选参考两类论文启发：
    - **0.1*fs (~100ms)**：对应微状态启发（EEG-Deformer）。
    - **Fs/4 (~250ms)**：对应MI节律更长时间窗（CTNet讨论MI解码窗口/超参敏感性）。  
把各分支输出 concat：得到 `F_t ∈ R^{B×Kf×C×T}`（这里Kf=分支数×每支filters）

**(2) Spatial Mixing（跨通道空间滤波）**  
用 depthwise/组卷积模拟空间滤波，典型就是 `(C,1)` 的空间卷积（EEG-Deformer明确用这个）。  
输出 `F_s ∈ R^{B×d×T}`（把通道维“揉”进特征维）

**(3) Patchify / Tokenize（关键！控制N）**  
用 **stride pooling 或 conv stride** 沿时间下采样成 tokens：

+ 参考CTNet：`P2` 决定 token size，并且有“过平滑 vs 噪声”的明确讨论与经验最优区间。   
我们的目标是让 **N大概在 15~30**（因为CTM每个tick都要attend一次，N太大会直接算力炸）。

实现方式建议两种开关：

+ `token_mode = "timepoint"`：像CTNet那样“每个时间点的feature channel作为token” 
+ `token_mode = "patch"`：像Conformer那样“pool成patch embedding当token”

最终得到：`tokens ∈ R^{B×N×d_kv}`

**(4) Positional Encoding（强烈建议保留开关）**  
EEG-Deformer是 **learnable position encoding直接加到tokens**。  
我们也做同样开关：`pos = learnable / sinusoidal / none`

---

## 4) 我建议你直接在代码里预留的“Tokenizer消融开关清单”
### Token来源相关（最重要）
+ `token_mode`: `patch` / `timepoint`
+ `N_target`: 目标token数（内部自动算stride/pool）
+ `pooling_P2` 或 `stride`: 直接控制token size（对照CTNet的结论做消融最有说服力） 

### 频带/时间尺度相关（MI任务高收益）
+ `temporal_ms_list`: 例如 `[100, 250]`（对应 0.1fs 与 fs/4 两派思路）
+ `fusion`: concat / sum / gated-sum（门控融合可作为“小创新点”）

### 空间结构相关（跨被试关键但要稳）
+ `spatial_mixer`: depthwise_conv / group_conv / (可选) lightweight-attn
+ `use_electrode_pos_emb`: on/off（先off也行，但建议保留）

### 归一化/非线性（EEG很敏感）
+ `norm`: BN / LN / GN
+ `act`: ELU / GELU  
（CTNet、EEG-Deformer都偏爱BN+ELU这一套）

---

## 5) 两个“容易踩坑但很好写进Methods规避”的点
1. **tokens数量N不要太大**：CTM是“多tick反复attend”，N一大，训练时间和显存会呈乘法爆炸（而且还更容易过拟合噪声）。CTNet对token size敏感性的讨论本质就在提醒这一点。
2. **Tokenizer里的任何“统计量”都要可控**：你前面已经决定trial-wise标准化，我们就坚持：Tokenizer内部不引入“跨trial/跨被试的归一化统计”。

好，我们把 **Tokenizer v1** 的“层级结构 + 默认超参 + 逐层张量形状 + 接到 CTM 的 kv_projector（d_kv→d_input）”一次性定下来，并且把所有关键点都做成**代码开关**，后面再做消融就只是改配置。

下面默认适配 **BCI-IV-2a：C=22, T=1000（2–6s@250Hz），目标 token 数 N≈20**。

---

## Tokenizer v1 总体接口
+ 输入：`X` 形状 **[B, C, T] = [B, 22, 1000]**
+ 输出：`tokens` 形状 **[B, N, d_kv] = [B, 20, 128]**（默认）
+ 可选输出：`tokens_mask`（默认 None，除非你后面做变长窗/丢段增强）

---

## 默认超参（v1 baseline）
### 核心维度
+ `d_kv = 128`（token embedding 维度）
+ `N_target ≈ 20`（目标 token 数）
+ `token_stride = 50`（因为 1000 / 50 = 20）
+ `token_pool_kernel = 50`（默认不重叠 patch；需要重叠就改 kernel>stride）

### 多尺度时间卷积（Multi-scale Temporal Conv）
+ `n_branches = 2`
+ `temporal_kernels = [25, 63]`
    - 25 点≈100ms，63 点≈250ms（两种典型时间尺度）
+ `F_per_branch = 8`（每个分支 8 个滤波器）
+ 合并后 `F_total = 16`

### 空间混合（Spatial Mixer）
+ `spatial_depth_multiplier = 2`（深度可分离空间卷积的倍率）
+ 空间卷积核：`(C, 1) = (22, 1)`（一次性跨通道）
+ 空间后通道数：`F_total * spatial_depth_multiplier = 32`

### 投影与归一化
+ `pointwise_proj`: 1×1 Conv 把 32 → 128
+ `conv_norm = BatchNorm2d`
+ `token_norm = LayerNorm(d_kv)`
+ `act = ELU`（更像 EEGNet/很多MI网络的稳定选择；当然可开关换 GELU）
+ `dropout = 0.25`（可开关）

---

## 逐层形状（从 raw 到 tokens）
约定：先把输入 reshape 成 2D conv 友好的形式。

### 0) Reshape
+ `X: [B, 22, 1000]`
+ `X2 = X.unsqueeze(1) → [B, 1, 22, 1000]`

### 1) Multi-scale Temporal Conv（两分支并联，只沿时间卷）
每个分支：

+ `Conv2d(in=1, out=8, kernel=(1, k), stride=(1,1), padding=(0, k//2), bias=False)`
+ 输出：`[B, 8, 22, 1000]`

拼接（concat on channel/filter dim）：

+ `F = concat(branches) → [B, 16, 22, 1000]`

### 2) Spatial Mixing（深度可分离：每个时间滤波器自己学一组空间滤波）
+ `Conv2d(in=16, out=16*2=32, kernel=(22,1), groups=16, bias=False)`
+ 输出：`[B, 32, 1, 1000]`

这一层相当于“每个 temporal feature 对应一组可学习的空间滤波”，非常贴 MI（C3/C4 周边模式）且实现简单。

### 3) Pointwise Projection（把通道数投到 d_kv）
+ `Conv2d(in=32, out=128, kernel=(1,1), bias=False)`
+ 输出：`[B, 128, 1, 1000]`

### 4) Tokenize（沿时间做 patch / stride pooling，把 1000 变成 ~20 个 token）
默认用 AvgPool：

+ `AvgPool2d(kernel=(1,50), stride=(1,50))`
+ 输出：`[B, 128, 1, 20]`

Rearrange 成序列 tokens：

+ `tokens = squeeze(dim=2).transpose(1,2)`
+ 得到：`tokens: [B, 20, 128]`

### 5) Position Encoding + Token Norm（可开关）
+ `pos_emb: [1, 20, 128]`（learnable，默认开）
+ `tokens = LayerNorm(tokens + pos_emb)`

---

## 连接到 CTM：kv_projector（d_kv → d_input）
CTM 的 cross-attn 需要 `K,V` 的最后一维等于 `d_input`（也就是 attention 的 `embed_dim`）。

### 默认建议（最省事、最稳定）
+ 设 `d_input = 128`
+ `kv_projector = Identity()` 或者 `Linear(128,128)`（保留一点自由度）
+ 这样 CTM 每个 tick 里：
    - `kv = kv_projector(tokens) → [B, 20, 128]`
    - `q_t`（来自同步表征投影）做成 `[B, 1, 128]`
    - `o_t = CrossAttn(q_t, kv, kv) → [B, 1, 128] → squeeze → [B, 128]`
    - `o_t` 回灌到 CTM 的 synapse（和 `z_t` concat）

### 你要的开关（建议在 config 里留这几个）
+ `d_kv`（Tokenizer 输出维度）
+ `d_input`（CTM attention 维度）
+ `kv_projector_type`: `"identity" | "linear" | "mlp"`
    - linear：`Linear(d_kv, d_input)`
    - mlp：`Linear(d_kv,d_input) + GELU + Linear(d_input,d_input)`（更强但更容易过拟合）

---

## 我建议你现在就写进代码里的 Tokenizer 开关清单
### Token/patch 相关
+ `token_stride`（默认 50）
+ `token_pool_kernel`（默认 50；>stride 就是重叠 patch）
+ `token_pool_type`: `"avg" | "max" | "conv_stride"`

### 多尺度时间卷积相关
+ `temporal_kernels`（默认 [25,63]）
+ `F_per_branch`（默认 8）
+ `branch_fusion`: `"concat" | "gated_concat"`（门控融合可作为小创新点）

### 空间混合相关
+ `spatial_depth_multiplier`（默认 2）
+ `spatial_mixer`: `"dw_conv(C,1)" | "group_conv" | "none"`

### 归一化/激活/正则
+ `conv_norm`: `"BN" | "GN" | "none"`
+ `token_norm`: `"LN" | "none"`
+ `act`: `"ELU" | "GELU"`
+ `dropout_p`

### 位置编码
+ `pos_emb`: `"learnable" | "sinusoidal" | "none"`

### CTM 接口
+ `d_kv`, `d_input`
+ `kv_projector_type`

---

好，那我们把 **Tokenizer v1（N≈20, d_kv=128）** 和 **CTM core** 的默认规模一起“配平”，形成一套可训练、可消融的 **v1 默认配置**（后续只改 config 开关，不改代码结构）。

下面这套默认值的目标是：**算力可控、机制完整（ticks + synchronization + NLM）、易做消融**。

---

## CTM v1 默认规模（与 Tokenizer 对齐）
### 1) 与 Tokenizer 的接口对齐
+ Tokenizer 输出：`tokens ∈ [B, N=20, d_kv=128]`
+ CTM cross-attn 维度：`d_input = 128`
+ `kv_projector`: 默认 `Identity()`（或开关切 `Linear(128→128)`）
+ 每个 tick 的 query：`q_t ∈ [B, 1, 128]`，对 `tokens` 做 cross-attn，得到 `o_t ∈ [B, 128]`（这就是 CTM 伪代码里的 `kv_projector(backbone(inputs))` 然后 `attn(q,kv,kv)` 那条线）。

### 2) CTM 核心超参（默认）
+ `D = 256`：CTM latent neurons 数（z 的宽度）
+ `T_internal = 12`：internal ticks（每个样本“思考”12步；CTM 的 internal tick 与数据维度解耦是核心机制之一）
+ `M_hist = 16`：pre-activation FIFO history 长度（给 NLM 用）
+ NLM（私有 MLP）：
    - `H = 64`（隐藏维）
    - 激活：GELU（或 ELU 开关）
+ Synapse（共享递归）：`MLP(D + d_input → D)`，两层（带 dropout 可选）
+ Synchronization pairs：
    - `K_action = 256`：用于生成 attention query 的 neuron-pairs 数
    - `K_out = 256`：用于分类 head 的 neuron-pairs 数  
同步表征来自历史活动的同步矩阵定义（`S_t = Z_t Z_t^T`），但我们用 pair 采样来避免 `O(D^2)`。

### 3) Attention 默认
+ `n_heads = 8`（embed_dim=128 可整除）
+ cross-attn：`Q` 长度=1（每 tick 一个 query），`KV` 长度=N=20（tokens）
+ 这一步的算力非常稳：每个样本每 tick 只 attend 20 个 token。

---

## 训练与输出（先定默认策略，开关做消融）
### 1) 每个 tick 都输出 logits（CTM 原生做法）
+ `logits_t = Head(S_out_t)`，最终得到 `logits ∈ [B, C, T_internal]`
+ Loss 默认：使用 CTM 的 “min-loss tick + max-certainty tick” 组合（让模型自然学到自适应计算倾向）  
（推理时可以取 `argmax(certainty)` 的 tick 或者最后 K 个 tick 平均，做开关。）

---

## 你现在就可以写进代码的“总配置”（建议 YAML/Dict）
```yaml
tokenizer:
  C: 22
  T: 1000
  d_kv: 128
  N_target: 20
  temporal_kernels: [25, 63]
  F_per_branch: 8
  spatial_depth_multiplier: 2
  token_pool:
    type: avg        # avg|max|conv_stride
    kernel: 50
    stride: 50
  norm: BN
  act: ELU
  dropout_p: 0.25
  pos_emb: learnable # learnable|sinusoidal|none

ctm:
  D: 256
  T_internal: 12
  M_hist: 16
  d_input: 128
  kv_projector: identity # identity|linear|mlp
  attn:
    n_heads: 8
  synapse:
    hidden: 512
    dropout_p: 0.1
  nlm:
    H: 64
    act: GELU           # GELU|ELU
  sync_pairs:
    K_action: 256
    K_out: 256
    decay: softplus_exp # exp(-softplus(r))
    pair_sampling: fixed_random # fixed_random|resample_each_run
  head:
    num_classes: 4
  tick_agg:
    train_loss: minloss_plus_maxcertainty
    infer: max_certainty # max_certainty|lastk_mean|mean_all
```

---

## 关键“代码开关”清单（你后面做消融就改它们）
### Tokenizer 侧
+ `temporal_kernels`（多尺度时间窗）
+ `token_pool.kernel/stride`（直接控制 N 与 token size）
+ `pos_emb`（none vs learnable）
+ `spatial_depth_multiplier`（空间混合强度）

### CTM 侧
+ `D`、`T_internal`、`M_hist`、`H`
+ `K_action`、`K_out`（同步对数）
+ `kv_projector`（identity/linear/mlp）
+ `tick_agg.infer`（多 tick 聚合策略）
+ `pair_sampling`（固定采样 vs 重新采样；固定更利于可复现）

---

下面把 **Tokenizer → CTM 的“拼接细节”**一次性讲清楚（按顶刊论文常见写法：先把接口/张量形状钉死，再把可选融合策略做成可控开关），并且每一步都对齐 EEG 的噪声/跨被试难点。

---

## 1) 总体数据流与接口契约（你后面写 Methods 就照这个）
**输入**：单 trial EEG，形状 `(B, C=22, T=1000)`（BCI-IV-2a，2–6s@250Hz）；常见做法是用 `[2,6]s` 并做 `[4,40]Hz` band-pass（很多 SOTA/强 baseline 都这么处理）。

**Tokenizer v1 输出**（我们已定）：

+ `tokens_raw`: `(B, N≈20, d_kv=256)`
+ `tokens`: `LayerNorm(tokens_raw)`（建议 LN 而不是 BN，避免跨被试/LOSO 时“batch 统计量混入”争议）

这一步的合理性：强论文里普遍是“卷积学局部时空特征 → pooling/patching 成 tokens → Transformer/Attention 学全局相关”。  
并且 token 数/大小对 MI 很敏感，CTNet 在 IV-2a 上给过“token size≈20附近更优”的经验（本质上支持你把目标 N 设在 20 左右）。

---

## 2) Tokenizer 如何接入 CTM 的 `kv_projector (d_kv → d_input)`
CTM 的标准写法是：`kv = kv_projector(backbone(inputs))`，再用 cross-attention：`o_t = Attention(q_t, kv, kv)`。

在我们这里，**backbone 就是 Tokenizer**，因此：

+ `kv = W_kv · tokens + b`
+ 形状：`tokens (B,N,256) → kv (B,N,d_input=128)`
+ 建议实现：`kv_projector = nn.Linear(256, 128, bias=True)`

这一步你可以理解为：Tokenizer 把 EEG 切成“可被注意力消费的 token”；`kv_projector` 只是把 token 的通道维投到 CTM 的注意力输入宽度 `d_input`，保持 CTM 内部统一尺度。

---

## 3) CTM 的 query `q_t` 从哪里来，以及形状怎么对齐
CTM 的核心是：**用“神经同步表示”生成 query**（action synchronization → `q_t`），再去 cross-attend tokens。

我们固定：

+ 同步向量：`S_action^t ∈ R^{D_action}`（比如 `D_action=64`）
+ `q_projector: R^{D_action} → R^{d_input}`（`Linear(64,128)`）
+ 形状对齐（给 MHA 用）：
    - `q = q_projector(S_action)` 得到 `(B,128)`
    - reshape 成 `(B,1,128)`
    - `o_t = MHA(q, kv, kv)` 得到 `(B,1,128)`（再 squeeze 成 `(B,128)`）

这就是 CTM 伪代码里 `q = q_projector(synch_a)`、`attn_out = attn(q, kv, kv)` 的 EEG 版落地。

---

## 4) 关键“拼接点”：`o_t` 如何注入 CTM synapse（给你 3 套可切换方案）
CTM 原始定义：  
[  
a_t = f_{\theta_{syn}}(\mathrm{concat}(z_t,\ o_t)) \in \mathbb{R}^{D}  
]  
并维护 FIFO 的 pre-activation history，再用每个 neuron 的私有 NLM 更新得到 (z_{t+1})。

### 方案 A（默认、最贴 CTM）：Concat → Synapse-MLP（UNet/MLP）
+ `u_t = concat([z_t, o_t])`，形状 `(B, D + d_input) = (B, 256+128)`
+ `a_t = synapse(u_t)`，输出 `(B, D)`
+ 优点：最忠实 CTM，结构清晰，后面写论文不容易被挑“改得不像 CTM”。
+ 风险：EEG token 噪声大时，`o_t` 可能“直接污染”synapse，导致动态不稳定（尤其 tick 多时）

**代码开关**：`fusion="concat"`

---

### 方案 B（强烈建议作为消融项）：FiLM 调制（用 o_t 调制 z_t 或 synapse 中间层）
FiLM 是顶会常用的“条件注入层”：(\mathrm{FiLM}(x)=\gamma(o)\odot x+\beta(o))。  
把它用在这里非常“数学可写、论文可讲”：

+ `gamma, beta = MLP(o_t)` → `(B,D)`
+ `z̃_t = gamma ⊙ LN(z_t) + beta`
+ `a_t = synapse(concat([z̃_t, o_t]))`

直觉对 EEG 很友好：**o_t 不再是硬拼接，而是“以条件的方式”去调制 latent**，更抗噪、更可控。

**代码开关**：`fusion="film"`，以及 `film_on="z" | "synapse_hidden"`

---

### 方案 C（非常 EEG 友好）：GRU-style 门控融合（把 o_t 当输入、z_t 当状态）
EEG 的低信噪比+跨被试漂移，本质上需要“选择性更新”。门控更新（GRU 思想）就是数学上很干净的做法：

+ `g_t = sigmoid(Wg·[z_t;o_t])`，形状 `(B,D)`
+ `h_t = tanh(Wh·[z_t;o_t])`
+ `z̄_t = (1-g_t)⊙z_t + g_t⊙h_t`
+ 再喂给 synapse：`a_t = synapse(concat([z̄_t, o_t]))`

优点：**当 o_t 不可靠时，门控会“少更新”**；对 LOSO 跨被试很常见地更稳。  
**代码开关**：`fusion="gated"`

---

## 5) Synchronization（S_action / S_out）怎么做才“严谨 + 高效 + 不怕审稿人”
CTM 同步的定义是：把 post-activation history 堆成 (Z^t)，做内积 (S^t = Z^t (Z^t)^\top)。  
但它是 (O(D^2))，所以必须 **采样 neuron pairs**：随机选 `(i,j)` 形成 `S_action^t` 与 `S_out^t`，并且建议包含一定数量的 **self-pairs (i,i)**，这样模型在需要时能退化成 snapshot 表示（CTM 文档明确说这是设计目的之一）。

### 我建议你在 EEG 上的“默认 pairing”
+ `pairing="random_with_self"`
+ `n_self = 8~16`（占 action/out 的一小部分）
+ 其余 pairs 随机（允许重叠）

### 同步的时间衰减（强烈建议保留）
CTM 用每个 pair 的可学习指数衰减 (r_{ij}) 来控制“更看重近期 tick 还是长期 tick”，并且给了递推式把复杂度降到 (O(D_{sub}))。  
这对 EEG 的意义：**你让模型自己学“需要多少内部思考步”与“用多长历史做同步”**，比你手工设定更容易在跨被试上泛化。

---

## 6) NLM（Neuron-Level Model）与实现要点：einsum + 稳定性开关
CTM 的 NLM 用 `einsum` 并行计算每个 neuron 的私有 MLP，非常关键（否则 D 个 MLP 会慢/难写）。  
你可以照它的两段 einsum 写（`(B,D,M) → (B,D,h) → (B,D)`），并做一个可选开关：

+ `nlm_norm="none" | "preact_ln"`  
CTM 报告里提到对 pre-activation history 做 LayerNorm 会更“周期性”，但整体性能通常更差（这正适合你做一个“论文里写得清楚的消融点”）。

---

## 7) 输出怎么取（tick 聚合）才能既贴 CTM、又适配分类
CTM 不是只用最后一个 tick，它提出了“最小 loss tick / 最大 certainty tick”等训练与聚合方式。  
我建议你实现 3 个推理模式作为开关：

+ `readout="last"`（最简单 baseline）
+ `readout="most_certain"`（贴 CTM 的思想）
+ `readout="certainty_weighted_mean"`（对 EEG 往往更稳）

（训练时可以用 CTM 的 `t1=argmin(L_t)` 与 `t2=argmax(C_t)` 组合 loss；推理时再切 readout。）

---

## 你现在就可以按这个骨架开工的“模块清单”（全是开关友好）
1. `EEGTokenizerV1(C=22,T=1000) -> (B,N=20,256)`
2. `KVProjector(256->128)`
3. `CTMCore(D=256, d_input=128, T_ticks=12, M_mem=16, D_action=64, D_out=64)`
4. `Fusion(fusion in {"concat","film","gated"})`
5. `PairSampler(pairing in {"random_with_self","dense","semi_dense"})`（先默认 random_with_self）
6. `SyncComputer(decay on/off, recursive on/off)`（先开递推）
7. `Readout(readout in {"last","most_certain","certainty_weighted_mean"})`

---

我们已经把 **CTM core 的“结构级设计”基本做完了**（到可以直接写成一个可训练的 PyTorch 模块），但**还没把所有“实现级”细节完全钉死**——主要是一些影响训练稳定性/算力/可复现的工程选择。下面我把“已完成 vs 还欠缺”的边界讲清楚，并给出一套我建议你们直接落地的 **v1 core 定稿**。

---

## 目前已经完成的 CTM core 设计（结构上已闭环）
这些是 CTM 的核心机制，我们都已经定了接口与实现路径：

1. **内部状态与递推**
+ 每个 tick 有神经元状态 `z_t ∈ R^{B×D}`，并维护 `pre-activation` 的 FIFO 历史 `A_t ∈ R^{B×D×M}`（给 NLM 用）。这就是 CTM 的 “synapse→pre-activation→历史→NLM→post-activation” 主链路。
1. **Synapse（共享递归）**
+ `a_t = Synapse([z_t; o_t]) -> R^{B×D}`，其中 `o_t` 来自 cross-attn（下一条）。CTM 的伪代码就是把 `z` 和 `attn_out` 拼起来喂进 synapse。
1. **NLM（每个 neuron 私有参数）**
+ 用 `einsum` 并行实现“每个 neuron 一套私有 MLP”，输入是该 neuron 的 `pre-activation history`，输出 `z_{t+1}`。这是 CTM 的关键创新之一。
1. **Synchronization 表征（action / out 两套）**
+ 用历史活动构建同步：理论上是 `S_t = Z_t Z_t^T`，但实现上做 **pair sampling** 避免 `O(D^2)`，得到 `S_action_t`（用于 query）与 `S_out_t`（用于输出）。
+ 并且我们明确要包含一小部分 **self-pairs**（能“退化回 snapshot”），这也是 CTM 设计意图之一。
1. **指数衰减/递推计算同步（高效且贴论文）**
+ 同步支持可学习时间衰减，并且可以递推维护（不必每 tick 扫一遍 history），CTM 文中也强调了这类递推/降复杂度思路。
1. **Cross-attention 读 tokens**
+ `q_t = proj(S_action_t)`，然后 `o_t = Attn(q_t, KV=tokens)`，并把 `o_t` 回灌进 synapse 形成闭环。CTM 伪代码就是这么写的。

到这一步为止，**CTM core 的结构已经是“完整且合理”的**。

---

## 还没“完全定稿”的部分（实现级细节，决定训练稳不稳/可复现）
这些不影响结构正确性，但会显著影响 EEG 上的训练稳定性与最终上限：

1. **Tokenizer 输出维度与 kv_projector 的最终口径**
+ 你之前认可的默认是 `tokens: [B,20,128]` + `d_input=128`，此时 `kv_projector=Identity()`最干净。  
（我建议就以这个作为 v1 定稿，别再摇摆。）
1. `**o_t**`** 注入 synapse 的融合方式**
+ 我们列了 `concat / FiLM / gated(GRU-style)` 三套（都合理），但 **默认要选一个**。对 EEG（噪声大、跨被试漂移）我更建议默认用 `gated` 或 `FiLM`，因为它能抑制不可靠的 `o_t`，防止 tick 多时动态发散。
1. **同步 pair 的采样策略与可复现**
+ 是否固定随机种子并缓存 pair 索引（推荐固定，保证 fold/seed 可复现）。
+ `K_action`、`K_out` 的默认值（我们建议 256/256 是合理起点）。
1. **初始化与数值稳定**
+ `z0`、`pre_hist0` 是学出来还是置零？
+ 是否对 `z_t` / `A_t` 加 LayerNorm（一般 LN 会更稳，但可能影响 CTM 的“动力学味道”，适合作为开关消融点）。
1. **tick 输出聚合与 certainty 的实现**
+ CTM 的 tick 级训练/聚合策略（min-loss tick、max-certainty tick、certainty-weighted mean 等）你们要在代码层面定默认，否则训练脚本无法统一。

---

## 我建议现在就“定稿”的 CTM core v1（可直接开工写代码）
如果你同意，我们就把 v1 默认定为：

+ `D=256, T_internal=12, M_hist=16, H=64`
+ `tokens: [B,20,128]`
+ `d_input=128, kv_projector=Identity()`
+ `fusion="gated"`（默认；`concat` 和 `film` 作为开关）
+ `pairing="fixed_random_with_self"`, `K_action=256`, `K_out=256`
+ `decay="learnable exp(-softplus(r))"`, `sync_update="recursive"`
+ `readout="most_certain"`（推理默认），训练 loss 先用“每 tick CE 的平均”或“min-loss+max-certainty”二选一做默认开关

这样就算你还没做任何“论文级创新”，模型也会非常“像样”：Tokenizer 提供稳定 tokens，CTM 用多 tick 迭代读取，内部用同步表征驱动 attention 和输出，NLM 给足动力学表达力。

---

如果你想让我回答得更“是/否”一点：  
**是的，CTM core 的核心结构我们已经设计完整了；但如果按工程落地标准来说，还差 5 类实现级默认决策（融合、pair、初始化、归一化、tick聚合）。**  
对，下一步就是 **Head / 多 tick 聚合策略**。你提到的那 “5 类实现级默认决策”里：

+ **属于 Head/聚合范畴的**：✅ **tick 聚合（readout）**（以及它依赖的 certainty 定义、是否 early-exit）
+ **不属于 Head 本身、但会强烈影响 Head 表现与稳定性的**：⚠️ **融合（fusion）/pair 采样/初始化/归一化**（它们在 CTM core 内，但会改变每个 tick 的 logits 质量与校准）

下面我把 **Head + 聚合**先“定成可直接写代码的 v1 默认方案”，同时把相关开关列清楚。

---

## 1) Tick-wise Head 设计（每个 internal tick 输出一次 logits）
### 1.1 从 synchronization 到 logits（最贴 CTM 的主干）
CTM 的标准形式是用输出同步向量 (S^{t}_{out}) 线性投影得到 (y_t)（logits）。  
**我们 v1 就保留这条主线**：

+ 计算：`S_out_t ∈ R^{Dout}`（来自 neuron-pair 同步 + 衰减）
+ Head：`logits_t = W_out S_out_t + b`，得到 `logits_t ∈ R^{C}`（C=4 类）

### 1.2 论文里常见的“2层全连接分类头”（作为可选开关）
很多高水平 EEG 分类网络在 Transformer/注意力后，用**两层 FC**做分类（简单但很强）：

+ EEG Conformer：明确用两层全连接做 classifier，并用交叉熵训练。
+ CTNet：也是“卷积模块 + Transformer encoder + 两层全连接分类器”的套路。

所以我们给一个开关：

+ `head_type="linear"`（默认，最贴 CTM）
+ `head_type="2fc"`：`Linear(Dout, h) + GELU/ELU + Dropout + Linear(h, C)`

EEG 的特点是 trial 数少、跨被试漂移大；“小头”往往比“大头”更不容易过拟合。默认 linear 很合理，2fc 留作提升上限的分支。

---

## 2) Certainty 定义（Head/聚合都要用）
CTM 里 certainty 用 **1 - normalized entropy**，并且配套了 loss/聚合策略。  
我们 v1 直接复用：

+ 先算概率：(p_t=\text{softmax}(\text{logits}_t))
+ 熵：(H_t=-\sum_c p_{t,c}\log p_{t,c})
+ 归一化：(\hat H_t = H_t / \log C)
+ certainty：(C_t = 1-\hat H_t)

---

## 3) 多 tick 聚合策略（Inference readout）
CTM 自己就强调：如何把多 tick 输出聚合成最终预测并不“显然”，并列了多种机制（most-certain、平均、certainty 加权平均等）。

结合 EEG 的低信噪比与跨被试漂移，我建议 v1 直接做成 4 个 readout 开关：

1. `readout="last"`：用最后一个 tick（baseline）
2. `readout="most_certain"`：取 (t^*=\arg\max_t C_t)，输出 `logits_{t*}`（贴 CTM 思路）
3. `readout="mean_logits"`：(\bar{\ell}=\frac{1}{T}\sum_t \ell_t)
4. ✅ `**readout="certainty_weighted"**`**（v1 默认）**：  
[  
w_t=\frac{\exp(\alpha C_t)}{\sum_\tau \exp(\alpha C_\tau)},\qquad \bar{\ell}=\sum_t w_t \ell_t  
]  
理由：EEG 往往某些 tick 会被噪声/伪迹“带偏”，加权平均能更稳；CTM 也讨论了“加权/平均”对表现与校准的影响。

另外可选加一个 **early-exit**（推理省算力，训练不变）：

+ `early_exit_tau=0.85`：当 (C_t>\tau) 就停止 ticks（CTM 用 certainty 阈值展示自适应计算潜力）。

---

## 4) 训练时的 tick loss（Deep supervision vs CTM 原生 loss）
你要的是跨被试稳健，我建议把训练也做成开关：

+ `tick_loss="ctm_t1t2"`：用 CTM 的 **min-loss tick + max-certainty tick**（核心公式与动机在 3.5 节）。
+ `tick_loss="mean_ce"`：所有 ticks 的 CE 平均（更像 deep supervision，通常更稳）
+ ✅ `tick_loss="hybrid"`（推荐作为 v1 默认候选）：  
[  
L = \lambda \cdot \text{MeanCE} + (1-\lambda)\cdot L_{\text{CTM}}  
]  
这样既保留 CTM 的“自适应计算”诱导，又避免 EEG 小数据时训练抖动。

---

可以。下面把你要的 **4 项实现级默认决策**写成“你拿去就能写 PyTorch 代码”的规格：**融合（fusion）/pair 采样/初始化/归一化**。我会尽量贴近 CTM 原始定义（便于后续对照），同时针对 EEG（低 SNR、LOSO 跨被试、小 batch）做稳健化处理。

---

## 1) Fusion：把 `o_t` 注入 CTM 的方式（默认：门控融合）
CTM 主干是把 attention 输出 `o_t` 与状态 `z_t` 拼接喂给 synapse：  
(a_t=f_{\theta_{syn}}(\text{concat}(z_t,o_t)))。  
但 EEG 噪声大，`o_t` 可能在某些 tick 不可靠，所以我建议默认用 **门控（GRU-style）**先“筛一遍”再进 synapse（让模型学会“少更新”）。

### Fusion 三种开关（都保留）
+ `fusion="concat"`：最贴 CTM（baseline）
+ `fusion="film"`：FiLM 调制（条件注入，常作为稳健增强）
+ ✅ `fusion="gated"`：门控融合（EEG 默认）

### 代码骨架（PyTorch）
```python
class Fusion(nn.Module):
    def __init__(self, D: int, d_input: int, mode: str = "gated"):
        super().__init__()
        self.mode = mode
        if mode == "film":
            self.film = nn.Sequential(
                nn.Linear(d_input, 2 * D),
                nn.GELU(),
                nn.Linear(2 * D, 2 * D),
            )
            self.z_ln = nn.LayerNorm(D)
        elif mode == "gated":
            self.gate = nn.Linear(D + d_input, D)
            self.cand = nn.Linear(D + d_input, D)

    def forward(self, z: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        # z: [B,D], o: [B,d_input]
        if self.mode == "concat":
            return torch.cat([z, o], dim=-1)   # [B, D+d_input]

        if self.mode == "film":
            gb = self.film(o)                 # [B,2D]
            gamma, beta = gb.chunk(2, dim=-1) # [B,D],[B,D]
            z_mod = gamma * self.z_ln(z) + beta
            return torch.cat([z_mod, o], dim=-1)

        if self.mode == "gated":
            u = torch.cat([z, o], dim=-1)
            g = torch.sigmoid(self.gate(u))   # [B,D]
            h = torch.tanh(self.cand(u))      # [B,D]
            z_bar = (1 - g) * z + g * h
            return torch.cat([z_bar, o], dim=-1)

        raise ValueError(self.mode)
```

你会发现这和 CTM 的“concat 进 synapse”主线完全兼容：只是把 concat 前的 `z` 做了一个“EEG 友好的选择性更新”。

---

## 2) Pair：同步表征的 neuron-pairs 采样与缓存（默认：固定随机 + self-pairs）
CTM 明确给了三种 pairing，并且在随机 pairing 时**刻意加入 (i,i) self-pairs**，保证模型必要时能退化成 snapshot 表示。  
同时 CTM 也给了“训练开始前采样 pairs、learnable 衰减 r 初始化为 0”的实现思路。

### 默认策略（v1）
+ `pairing="fixed_random_with_self"`
+ `Daction = 256`, `Dout = 256`
+ `nself = 16`（占一小部分即可）
+ action/out **各自一套** pairs（可以允许 overlap；CTM 随机 pairing 允许 overlap）

### 代码骨架
```python
def sample_pairs(D: int, Dsub: int, nself: int, generator: torch.Generator):
    assert 0 <= nself <= Dsub
    # self-pairs: pick unique neurons for diagonals
    self_idx = torch.randperm(D, generator=generator)[:nself]
    left_self = self_idx
    right_self = self_idx

    # random pairs (can overlap)
    n_rand = Dsub - nself
    left = torch.randint(0, D, (n_rand,), generator=generator)
    right = torch.randint(0, D, (n_rand,), generator=generator)

    idx_left = torch.cat([left_self, left], dim=0)   # [Dsub]
    idx_right = torch.cat([right_self, right], dim=0)
    return idx_left, idx_right

class PairBank(nn.Module):
    def __init__(self, D: int, Daction: int, Dout: int, nself: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        aL, aR = sample_pairs(D, Daction, nself, g)
        oL, oR = sample_pairs(D, Dout,    nself, g)
        self.register_buffer("act_left", aL)
        self.register_buffer("act_right", aR)
        self.register_buffer("out_left", oL)
        self.register_buffer("out_right", oR)
```

这样每个 fold/seed 完全可复现，也符合 CTM “pair 在初始化阶段预先选好”的范式。

---

## 3) 初始化：`z_init` 与 `pre_acts_history_init`（默认：零初始化，但保留 learnable 开关）
CTM 原文的设定是：**初始 z 与初始 pre-activation history 是可学习参数**。 并且 Listing 里就是 `z_init`、`pre_acts_history_init` 两个 Parameter。

但在 EEG/LOSO 小数据场景，可学习初始状态有时会变成“被试/会话偏置容器”，更容易过拟合。所以我建议：

### v1 默认
+ `init_mode="zeros"`：`z0=0, A0=0`（训练最稳、最不容易引入偏置）
+ 开关保留：
    - `init_mode="learnable"`：完全对齐 CTM（如果你后面做大规模多数据集预训练，这个往往更好）
    - `init_mode="learnable_noise"`：在 learnable 基础上加很小高斯噪声（打破对称）

### 代码骨架
```python
class CTMInit(nn.Module):
    def __init__(self, D: int, M: int, mode: str = "zeros"):
        super().__init__()
        self.mode = mode
        if mode == "learnable" or mode == "learnable_noise":
            self.z_init = nn.Parameter(torch.zeros(D))
            self.A_init = nn.Parameter(torch.zeros(D, M))
        else:
            self.register_buffer("z_init", torch.zeros(D))
            self.register_buffer("A_init", torch.zeros(D, M))

    def make(self, B: int, device=None, dtype=None):
        z = self.z_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1).contiguous()     # [B,D]
        A = self.A_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous() # [B,D,M]
        if self.mode == "learnable_noise":
            z = z + 0.01 * torch.randn_like(z)
            A = A + 0.01 * torch.randn_like(A)
        return z, A
```

---

## 4) 归一化：LN 放哪里（默认：LN on z + LN on o + 可选 LN on pre-acts history）
### 为什么 EEG 更推荐 LN（而不是 BN）
LOSO 下 batch 往往小、且跨被试分布差异大；**BN 依赖 batch 统计量**，对跨被试泛化和可复现都不友好。Transformer/注意力体系里也普遍用 LN（稳定且不依赖 batch 统计）。

### v1 默认的 LN 放置（最稳、最少争议）
1. `o_t`（attention 输出）做 LN：防止某次 tick 的注意力输出幅值爆炸
2. `z_t` 做 LN：让 NLM/synapse 接收到的状态尺度稳定
3. **不开**`pre_acts_history` 的 LN（默认关）
    - CTM 明确提到：对 pre-activation histories 做 LN 会让“周期性更早出现”，但性能一般更差，是一个可选项。

另外，CTM 在某些设置（RL）里也显式使用了 LayerNorm 与门控结构（GLU）来提高训练稳定性，这个经验对 EEG 也很有借鉴意义。

### 代码骨架
```python
class CTMNorm(nn.Module):
    def __init__(self, D: int, d_input: int, ln_on_preact_hist: bool = False):
        super().__init__()
        self.z_ln = nn.LayerNorm(D)
        self.o_ln = nn.LayerNorm(d_input)
        self.ln_on_preact_hist = ln_on_preact_hist
        if ln_on_preact_hist:
            # LN over the memory dimension is unusual; we instead LN over feature dim per time-slice
            # We'll implement LN over D for each memory step by reshaping.
            self.A_ln = nn.LayerNorm(D)

    def norm_z(self, z):  # [B,D]
        return self.z_ln(z)

    def norm_o(self, o):  # [B,d_input]
        return self.o_ln(o)

    def norm_A(self, A):  # [B,D,M]
        if not self.ln_on_preact_hist:
            return A
        # apply LN over D for each memory index: treat A as [B*M, D]
        B, D, M = A.shape
        x = A.permute(0, 2, 1).reshape(B * M, D)
        x = self.A_ln(x)
        return x.reshape(B, M, D).permute(0, 2, 1)
```

---

## 这四项拼进你的 CTM tick（“不歧义”的顺序）
按 CTM 的主链路（attention→concat→synapse→FIFO→NLM→sync）来写，推荐的 tick 内顺序是：

1. `kv = tokens`（Tokenizer 已 LN；若 kv_projector=Identity 就直接用）
2. `q_t = q_proj(S_action_t)` → `o_t = attn(q_t, kv, kv)`
3. `o_t = LN(o_t)`
4. `z_t = LN(z_t)`
5. `u_t = fusion(z_t, o_t)`（concat/film/gated）
6. `a_t = synapse(u_t)` 生成 pre-activation
7. 更新 FIFO：`A_t = cat(A_{t-1}[:,:,1:], a_t)`
8. （可选）`A_t = norm_A(A_t)`
9. `z_{t+1} = NLM(A_t)`（einsum 版）
10. 更新同步（用 pairs + 递推指数衰减）：见 CTM 递推公式 (16)(17)，可做到 O(Dsub) 每 tick。

---

下面把你要的 **“pair 同步的递推实现（α/β 递推）”**写成一个独立模块：它严格对应 CTM Appendix H 的公式 (13)–(17)：先定义带指数衰减的同步 (S^t_{ij})，再用两条一阶递推维护 (\alpha^t_{ij},\beta^t_{ij})，做到每 tick 只需 (O(D_{sub})) 更新。

---

## 1) 数学口径（你代码里要实现的就是这三行）
CTM 给的带衰减同步（对一个 pair (i,j)）是：  
[  
S^t_{ij}=\frac{\sum_{\tau=1}^{t} e^{-r_{ij}(t-\tau)} z^\tau_i z^\tau_j}{\sqrt{\sum_{\tau=1}^{t} e^{-r_{ij}(t-\tau)}}}  
]  
并定义辅助序列 (\alpha^t_{ij},\beta^t_{ij})，使得 (S^t_{ij}=\alpha^t_{ij}/\sqrt{\beta^t_{ij}})，且满足递推：  
[  
\alpha^{t+1}_{ij} = e^{-r_{ij}}\alpha^t_{ij} + z^{t+1}_i z^{t+1}__j,\quad  
__\beta^{t+1}__{ij} = e^{-r_{ij}}\beta^t_{ij} + 1.  
]  


另外，论文 Listing 3 里也说明了：(r) 初始化为 0（即无衰减），并在实践中用递推来降算。

---

## 2) PyTorch 模块：对一组 pairs 的递推同步（支持 batch）
### 设计要点（EEG/LOSO 友好）
+ `r_ij ≥ 0`：用 `softplus(raw_r)` 保证非负，再做 `decay = exp(-r)`（严格对应公式里 (e^{-r_{ij}})）。
+ `left/right` pairs 作为 `buffer` 固定（可复现）；支持 self-pairs（(i,i)）。CTM 的 random pairing 明确建议包含 self-pairs 来保证能恢复 snapshot。
+ 递推状态 `alpha/beta` 每个 forward 里初始化并随 tick 更新（不跨 batch 持久化，避免泄露/状态污染）。

### 代码
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursivePairSynchronizer(nn.Module):
    """
    Maintains exponentially-decayed synchronization for a *subsampled* set of neuron pairs.

    z_t: [B, D] post-activations at internal tick t
    pairs: left/right index tensors, each [Dsub]
    state: alpha, beta each [B, Dsub]

    Implements Appendix H recurrences:
      alpha_{t+1} = exp(-r) * alpha_t + z_{t+1,i} z_{t+1,j}
      beta_{t+1}  = exp(-r) * beta_t  + 1
      synch_t     = alpha_t / sqrt(beta_t)
    """
    def __init__(self, D: int, Dsub: int, left_idx: torch.Tensor, right_idx: torch.Tensor,
                 eps: float = 1e-8):
        super().__init__()
        assert left_idx.shape == (Dsub,)
        assert right_idx.shape == (Dsub,)
        self.D = D
        self.Dsub = Dsub
        self.eps = eps

        self.register_buffer("left", left_idx.long())
        self.register_buffer("right", right_idx.long())

        # raw_r initialized to 0 => r=softplus(0)~0.693; if you want exactly 0 decay at init,
        # set raw_r to a large negative value (e.g. -10). CTM listing says r initialized as zeros. 
        # We'll follow CTM and keep raw_r = 0, but expose a helper below.
        self.raw_r = nn.Parameter(torch.zeros(Dsub))

    def set_no_decay_init(self):
        # Make r ~= 0 at init: softplus(-10) ~ 0
        with torch.no_grad():
            self.raw_r.fill_(-10.0)

    def init_state(self, z0: torch.Tensor):
        """
        Initialize alpha/beta from an initial z (t=1 base case).
        alpha_1 = z1_i z1_j, beta_1 = 1.
        """
        prod0 = z0[:, self.left] * z0[:, self.right]  # [B, Dsub]
        alpha = prod0
        beta = torch.ones_like(alpha)
        synch = alpha / torch.sqrt(beta + self.eps)
        return alpha, beta, synch

    def step(self, z_next: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """
        Update alpha/beta given z_{t+1}.
        """
        r = F.softplus(self.raw_r)              # [Dsub], ensure r>=0
        decay = torch.exp(-r).unsqueeze(0)      # [1, Dsub]

        prod = z_next[:, self.left] * z_next[:, self.right]  # [B, Dsub]
        alpha = decay * alpha + prod
        beta = decay * beta + 1.0
        synch = alpha / torch.sqrt(beta + self.eps)
        return alpha, beta, synch
```

注：CTM 伪代码里 `r` 初始化为 0（无衰减）对应的是“直接把 r 当作衰减率”时的直觉。  
但如果你用 `r=softplus(raw_r)`，那 `raw_r=0` 会得到 `r≈0.693`（有一点衰减）。所以我给了 `set_no_decay_init()` 这个小工具，让初始真接近 0 衰减（更贴原意）。工程上我更推荐这种“可控初始化”。

---

## 3) Action / Output 两套同步一起封装（你 CTM 里直接用）
```python
class CTMSyncModule(nn.Module):
    def __init__(self, D: int,
                 Daction: int, act_left: torch.Tensor, act_right: torch.Tensor,
                 Dout: int, out_left: torch.Tensor, out_right: torch.Tensor,
                 eps: float = 1e-8):
        super().__init__()
        self.action = RecursivePairSynchronizer(D, Daction, act_left, act_right, eps=eps)
        self.output = RecursivePairSynchronizer(D, Dout, out_left, out_right, eps=eps)

    def set_no_decay_init(self):
        self.action.set_no_decay_init()
        self.output.set_no_decay_init()
```

---

## 4) 在 CTM forward 里怎么用（最清晰的 tick 顺序）
CTM 在每个 internal tick 都要得到：

+ `synch_a`：用于投影 query（attention）
+ `synch_o`：用于投影 logits/output（head）

你可以这样组织（示意）：

```python
# 假设 z 已经初始化好: [B,D]
alpha_a, beta_a, synch_a = sync.action.init_state(z)
alpha_o, beta_o, synch_o = sync.output.init_state(z)

for t in range(T_internal):
    # 1) q from synch_a, attend to tokens -> o_t
    q = q_proj(synch_a).unsqueeze(1)          # [B,1,d_input]
    o_t, _ = attn(q, kv, kv)                  # [B,1,d_input]
    o_t = o_t.squeeze(1)                      # [B,d_input]

    # 2) synapse + NLM -> new z
    u = fusion(z, o_t)                        # [B,D+d_input] (或其他)
    a_t = synapse(u)                          # [B,D]
    A = torch.cat([A[:, :, 1:], a_t.unsqueeze(-1)], dim=-1)  # FIFO: [B,D,M]
    z = nlm(A)                                # [B,D]

    # 3) 更新同步（递推）得到下一步的 synch_a/synch_o
    alpha_a, beta_a, synch_a = sync.action.step(z, alpha_a, beta_a)
    alpha_o, beta_o, synch_o = sync.output.step(z, alpha_o, beta_o)

    # 4) 输出 logits（head）
    logits_t = out_proj(synch_o)              # [B,C]
```

这条线和 CTM 的描述是一致的：同步用于产生 query 与输出，同时衰减同步的“高效递推”来自 Appendix H。

---

建议：只要加一条“跨被试数学约束”，论文就从“换骨架”变成“解决科学问题”：

+ **Subject-adversarial（域对抗）约束**：在 `z_t`（或 tokens）上加一个“预测被试 ID”的辅助头，用 GRL 反向梯度让主干学到“被试不可分”的表示。
+ 或者 **跨被试 supervised contrastive**：同类不同被试拉近、异类拉远（比纯 CE 更直接瞄准跨被试）。

这样你就能非常硬地回答审稿人：“你不仅用了 CTM，你还在优化 _subject-invariant representation_。”

这个建议**非常合理**，而且在审稿语境里属于“从换骨架到在优化跨被试不变性”的关键一步：你把跨被试问题明确写成 **subject-invariant representation learning**，而不是“我换了个 backbone 刚好更好”。

但实现时确实有一些容易翻车的点（尤其在 LOSO、EEG 小数据、CTM 多 tick 结构下）。我分两条讲：**域对抗（GRL）****和****跨被试 supervised contrastive**，再给你一个推荐落地方案。

---

## 1) Subject-adversarial（域对抗 + GRL）是否合理？
合理，原因是它直接对应一个清晰的数学目标：

+ 任务损失：让表征可区分类别（MI 四分类）
+ 对抗损失：让表征**无法区分被试 ID**

通常写成（概念上）：  
[  
\min_{\theta_f,\theta_y}\max_{\theta_s}\ \mathcal L_{\text{cls}}(y, f_y(f_\theta(x)))\ -\ \lambda,\mathcal L_{\text{subj}}(s, f_s(f_\theta(x)))  
]  
用 GRL 实现“最大化 subject loss / 最小化 class loss”在同一条反传里完成。

### 实现时要注意的坑
### A. 在 LOSO 里“subject head 预测的类别数”怎么定义？
+ 只能用**训练被试**的 subject ID（比如 8 个被试），subject head 输出维度就是 `n_train_subjects`。
+ **不要**在训练中使用测试被试任何数据去更新 subject head（否则变 transductive 了）。

这不影响“泛化到新被试”的目标：我们用“对训练被试不可分”作为 proxy，推动学到更稳健的表征。

### B. GRL 的 λ 不能一上来就大（会把分类能力也抹掉）
EEG 的类判别特征本来就弱，域对抗如果太强，会出现“表征被洗平 → 分类也学不动”。

工程上建议：

+ `λ_adv`**warm-up**：从 0 线性/非线性升到目标值（比如前 20–40% epoch 逐步升）
+ 或者做 DANN 常见的 schedule（随训练进度增长）

### C. 被试与类别的“伪相关”会让对抗训练误伤
如果训练集里某些被试对某些类的 trial 分布不均衡，subject classifier 会间接利用类别信息；GRL 会反过来惩罚这些“类别相关特征”，导致主任务掉点。

你需要：

+ **按被试×类别分层采样**（尽量让每个 batch 的类别、被试都更均衡）
+ 或者 subject loss 做 re-weight（按 subject 频率）

### D. 在 CTM 多 tick 下，约束加在哪里？
这是你们的独特难点，也是优势：你可以让“每一步思考都更不带被试信息”。

推荐两种稳妥选项（都留开关）：

1. **加在 tokens 上（更像“前端不变”）**  
`rep = mean_pool(tokens)` 或 `CLS-like token`  
优点：推动 Tokenizer 学到被试不变；缺点：可能过早抹掉对 MI 有用的空间模式。
2. ✅ **加在 CTM 的状态上（更贴“思考表征不变”）**  
`rep_t = z_t`（或 LN 后的 `z_t`）  
做法：对每个 tick 都计算一个 subject loss，然后平均：  
[  
\mathcal L_{\text{subj}}=\frac{1}{T}\sum_t \mathrm{CE}(h_s(\text{stop?}(rep_t)), s)  
]  
优点：与 CTM 结构最一致；而且“多 tick 平均”会比只用最后一步更稳。

---

## 2) 跨被试 supervised contrastive 是否合理？
也合理，而且很多时候比域对抗**更直接、更稳定**：你明确规定“同类不同被试要靠近”，而不是只说“被试不可分”。

一个你想要的正样本定义是：

+ **正样本**：同类别 (y) 且被试不同 (s\neq s')
+ **负样本**：类别不同（不管被试）

这正中跨被试痛点。

### 实现时要注意的坑
### A. batch 里必须“同类多被试”同时出现，否则对比损失没法算
这是最常见的工程坑。你需要一个 sampler，保证每个 batch 至少：

+ 每类有 ≥2 个不同被试的样本（越多越好）  
否则很多 anchor 没有正样本，loss 退化。

解决方案：

+ 做 **class-aware + subject-aware batch sampler**
+ 或者用 **memory bank / queue** 存储最近的 embedding（跨 batch 也能找到正样本）

### B. 用什么 embedding 做对比？
别用 logits。建议用“分类前的表征”，例如：

+ `e_t = proj(z_t)`（CTM 内部表征）
+ 或 `e = proj(mean_pool(tokens))`（前端表征）  
并 L2 normalize。

CTM 多 tick 的做法建议：

+ 用最后几步的平均（或 certainty-weighted）embedding：`e = Σ w_t proj(z_t)`  
这样更稳，也更符合“思考后表征”。

### C. 增强怎么配合？
你们已经打算用 S&R（同类拼接），它天然生成“同类视图”。一个很强的组合是：

+ 每个原始样本生成一个增强视图（S&R 或轻噪声），形成**同一 trial 的两视图**
+ 再叠加“同类不同被试”的正样本  
这样对比损失会更充实。

---

## 3) 选哪条作为 v1？我建议的落地顺序
在 EEG + LOSO + CTM 多 tick 下，我更推荐：

1. **先上 supervised contrastive（跨被试正样本定义）**作为主“数学约束”  
稳定、可控、解释强。
2. 再把 **subject-adversarial（GRL）作为小权重辅助项**（可开关）  
用 warm-up 的 λ_adv，避免把特征洗没了。

最终总损失建议留成开关：  
[  
\mathcal L = \mathcal L_{\text{cls}} + \lambda_{\text{con}},\mathcal L_{\text{supcon}} + \lambda_{\text{adv}},\mathcal L_{\text{subj}}  
]

---

## 4) 你在模型实现里需要新增的模块与开关（最实用）
### 必备模块
+ `SubjectHead`: MLP(rep_dim → n_train_subjects)
+ `GRL(lambda_adv)`：梯度反转层
+ `ProjHead`: MLP(rep_dim → d_contrast) + normalize
+ `SupConLoss(temperature)`

### 关键开关（建议 config 化）
+ `rep_for_invariance`: `"tokens_mean" | "z_last" | "z_mean" | "z_certainty_weighted"`
+ `adv_on`: on/off
+ `adv_lambda_schedule`: linear/cosine/warmup_steps
+ `supcon_on`: on/off
+ `supcon_positive_rule`: `"same_class_diff_subject" | "same_class_any_subject"`
+ `batch_sampler`: `"subject_class_balanced"`（强烈建议）
+ `tick_level_loss`: `"all_ticks_mean" | "last_tick" | "topk_ticks"`

---

可以。下面我把 **“把约束插到你们 CTM v1 里”**做成一套可直接照着写代码的方案：**插入点选哪里、怎么聚合多 tick、怎么做 sampler、训练 step 伪代码、以及哪些地方最容易踩坑**。我也会沿用顶会/顶刊常用的两条路线：**Domain-Adversarial (DANN/GRL)** 和 **Supervised Contrastive**，但把细节改到更适合 EEG（低 SNR、小 batch、LOSO 跨被试）。

---

# 1) 约束加在哪里最合理
你们模型有三种“可用表征”：

1. **tokens**：`tokens ∈ [B,N, d_kv]`（来自 Tokenizer）
2. **CTM attention 输出**：`o_t ∈ [B,d_input]`
3. ✅ **CTM 内部状态**：`z_t ∈ [B,D]`（每个 tick 一个）

对 EEG 跨被试，我建议 v1 默认把“跨被试不变性”加在 `**z_t**`** 上**，原因很实用：

+ tokens 更靠近原始信号，包含大量被试/阻抗/噪声统计，强行做 subject-invariant 容易把 MI 信息也洗掉。
+ `z_t` 是 CTM “思考后”的表征，更适合对齐“任务语义不变性”。

---

# 2) 多 tick 怎么聚合成一个“用于约束”的 representation
你们 CTM 每个 tick 都有 logits，因此可以计算每 tick 的 certainty（CTM 用的是 1-归一化熵）。  
我建议实现一个通用聚合器 `aggregate_rep(z_ticks, logits_ticks)`，作为 **adv head / contrastive head** 的输入。

### v1 默认：certainty-weighted 聚合（抗噪更稳）
+ 计算每 tick 的权重 (w_t)（softmax over certainty）
+ `rep = Σ_t w_t * LN(z_t)`  
这种做法对 EEG 很友好：遇到噪声 tick（不确定）会自动降权。

### 开关（都实现，后面消融用）
+ `rep_mode="last"`：`rep=z_T`
+ `rep_mode="mean_last_k"`：平均最后 K 步（K=3）
+ ✅ `rep_mode="certainty_weighted"`：默认

---

# 3) Subject-adversarial（GRL）怎么插进来
## 3.1 结构
+ `rep_adv = aggregate_rep(...)`，形状 `[B, D]` 或投影后 `[B, d_adv]`
+ `rep_adv_grl = GRL(rep_adv, λ_adv)`
+ `subj_logits = SubjectHead(rep_adv_grl)`，输出维度 = **训练折的 subject 数**（LOSO 下为 8）

测试时 **完全不需要 subject head**，它只在训练时提供反向约束。

## 3.2 关键实现注意点（非常重要）
1. **subject head 的类别数是训练被试数（8）**
    - 把训练被试 ID 映射到 `0..S-1`（S=8）。
    - 测试被试不参与训练，不存在 “第9类”。
2. **λ_adv 必须 warm-up**
    - 否则很容易“表征被洗平 → CE 学不动”。
    - 实现：`λ_adv = λ_max * schedule(progress)`（线性或 sigmoid）
3. **batch 里 subject/类分布要尽量均衡**
    - 否则 subject classifier 借助类别偏差投机，GRL 会误伤任务特征。

---

# 4) 跨被试 supervised contrastive 怎么插进来
## 4.1 正负样本定义（你提出的那句“同类不同被试拉近”就是核心）
对一个 anchor (i)：

+ 正样本：`same class AND different subject`
+ 负样本：`different class`（subject 不管）

工程上我强烈建议你 **再加一个永远存在的正样本**：同一个样本的“增强视图”（view1/view2），否则 batch 里凑不齐“同类不同被试”时 loss 会大量退化。

也就是 positives 包含两类：

+ `pos_type_1`: 同一 trial 的两视图（最稳定）
+ ✅ `pos_type_2`: 同类不同被试（跨被试目标）

## 4.2 用哪个 embedding 做对比
+ `rep_con = aggregate_rep(...)`（同 adv 一套聚合）
+ `e = normalize(ProjHead(rep_con))`（L2 normalize）
+ 用 SupCon loss（温度 τ 可配）

## 4.3 最容易踩的坑
1. **batch 里必须保证“同类多被试”**
    - 否则跨被试 positives 不存在。
    - 解决：做 subject-class balanced sampler（下面给你）。
2. **contrastive 头不要太大**
    - EEG 小数据容易过拟合。ProjHead 1–2 层足够。

---

# 5) 你需要的 Sampler（否则 SupCon 很难工作）
建议一个简单但有效的 **(class,subject)-balanced sampler**：

+ 每个 batch 选择 `S` 个被试（例如 4 个）
+ 对每个被试，采样每个类 `m` 个样本（例如每类 2 个）
+ 那么 batch size = `S * C * m`（C=4 类）

例如 `S=4, m=2` ⇒ batch=32  
这个 batch 里每个类都有多个被试，SupCon 才有跨被试正样本。

LOSO 时你的 dataset 只包含训练被试，sampler 只在训练被试上运作，天然不泄露。

---

# 6) 训练 step 伪代码（你照着写就能跑）
下面假设：

+ `model(x)` 返回 `logits_ticks: [B,C,T]` 和 `z_ticks: [B,T,D]`（你在 CTM forward 里把每 tick 的 z 存起来即可）
+ `augment_sr(x)` 生成增强视图（S&R 只训练用）
+ `aggregate_rep(z_ticks, logits_ticks)` 实现 rep_mode（默认 certainty_weighted）

```python
# batch: x [B,C,T], y [B], subj [B]  (subj 已经映射到 0..S-1)
x1 = x
x2 = augment_sr(x)          # 训练才做；val/test 不做

logits1, zticks1 = model(x1)  # logits1: [B,C,Tticks], zticks1:[B,Tticks,D]
logits2, zticks2 = model(x2)

# ---- classification loss (tick-wise deep supervision) ----
# v1 稳妥：mean over ticks
loss_cls = 0
for t in range(Tticks):
    loss_cls += CE(logits1[:,:,t], y)
loss_cls = loss_cls / Tticks

# ---- aggregate reps for invariance losses ----
rep1 = aggregate_rep(zticks1, logits1)   # [B,D]
rep2 = aggregate_rep(zticks2, logits2)   # [B,D]

# ---- supervised contrastive ----
e1 = normalize(proj_head(rep1))          # [B,d_con]
e2 = normalize(proj_head(rep2))          # [B,d_con]
E  = concat([e1,e2], dim=0)              # [2B,d_con]
Y  = concat([y,y], dim=0)                # [2B]
S  = concat([subj,subj], dim=0)          # [2B]

# positives mask:
# 1) same instance cross-view: i<->i+B
# 2) same class different subject: (Y_i==Y_j) & (S_i!=S_j)
loss_con = SupCon(E, labels=Y, subjects=S, include_same_instance=True)

# ---- subject adversarial ----
rep_adv = rep1                            # 也可以用(rep1+rep2)/2
subj_logits = subj_head(GRL(rep_adv, lambda_adv))
loss_adv = CE(subj_logits, subj)

# ---- total ----
loss = loss_cls + lam_con * loss_con + lam_adv * loss_adv
loss.backward()
opt.step()
```

### λ_adv schedule（必须加）
```python
progress = global_step / total_steps
lambda_adv = lam_adv_max * warmup(progress)   # 例如前 30% 线性到 1，再保持
```

---

# 7) 推荐的 v1 默认开关（你写 config 时直接照抄）
+ `rep_mode: certainty_weighted`
+ `tick_loss: mean_ce`（先稳；后面再试 CTM 的 t1/t2 loss）
+ `supcon_on: true`
+ `supcon_tau: 0.07`
+ `supcon_pos_rule: same_class_diff_subject + same_instance`
+ `adv_on: true`（但）
+ `adv_lambda_max: 0.1`（先小，warm-up 到 0.1）
+ `adv_warmup: 0.3`（前 30% steps 逐步加）
+ `sampler: subject_class_balanced (S=4, m=2)`（示例）

---

# 8) 你实现时最要防的 3 个“隐性问题”
1. **不均衡采样导致“对抗误伤”**  
subject head 如果能靠类别分布投机，GRL 会把类别信息也压掉。必须靠 sampler 或 reweight 抑制。
2. **batch 太小 → SupCon 没正样本**  
没有跨被试 positives 会导致 loss 退化。必须 sampler 或 memory queue。
3. **约束加得太早/太重**  
EEG 特征弱，先让 CE 学到基本可分，再逐步增强 invariance（尤其是 GRL）。

---





