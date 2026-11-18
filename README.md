# 高噪声观测下的稀疏信号恢复

给定已知字典 $A$ 和带噪观测 $y$，估计稀疏信号 $x$。五个条件为 40、20、10、5 dB 和无噪声（SNR 为无穷），每个条件独立训练、验证和测试。EBT-LISTA、HLISTA、ALISTA、ELISTA 各做逐层联合微调和端到端，NA-ALISTA 使用全深度端到端反传。种子 0 共 45 个模型。

## 数据协议

| 项目 | 首版值 |
|---|---|
| 字典 | $A\in\mathbb R^{250\times500}$，独立标准高斯元素，逐列归一化至单位二范数。三个数据划分共用同一张字典 |
| 稀疏信号 | 每个样本从 500 个位置均匀无放回抽取 50 个位置。非零系数独立服从标准正态分布，保留自然幅值 |
| 监督标签 | 无噪声的稀疏系数 $x$ |
| 固定数据量 | 训练 10,000、验证 1,000、测试 2,000 条独立信号。每条信号配四个带噪观测和一个无噪声观测。NA-ALISTA 的在线训练采样见下文 |
| 随机性 | 数据种子 20261118。字典和各划分使用独立随机流，每个划分的支撑、幅值、噪声各自取样 |
| 模型输入 | 同一 $A$ 和已保存的 $y$。SNR 标签用于分组评估，模型前向仅接收观测及计算深度 |

令 $m$ 为观测维数，$n$ 为信号维数，$k$ 为非零元素数，$r$ 为目标观测 SNR（dB）。对每条干净信号独立抽取一个 $g\sim\mathcal N(0,I_m)$，在四档 SNR 之间复用。

$$
y_r=Ax+\sigma_r g,\qquad
\sigma_r=\sqrt{k/m}\,10^{-r/20}.
$$

均匀支撑和单位方差系数给出 $\mathbb E[xx^\top]=(k/n)I_n$。由于 $\|A\|_F^2=n$，有 $\mathbb E\|Ax\|_2^2=k$，且 $\mathbb E\|\sigma_r g\|_2^2=m\sigma_r^2$。因此两者的期望功率比是 $10^{r/10}$。各档采用各自固定的噪声方差，单条样本的实际 SNR 会波动。同一划分内，不同档位之间的误差比较可以按干净信号配对。

| 目标 SNR（dB） | 噪声标准差 $\sigma_r$ | 首版测试集实测 SNR（dB） |
|---:|---:|---:|
| 40 | 0.004472 | 40.023 |
| 20 | 0.044721 | 20.023 |
| 10 | 0.141421 | 10.023 |
| 5 | 0.251487 | 5.023 |

实测值按整个测试集的信号总能量除以噪声总能量，再取 $10\log_{10}$。[数据配置](experiment.json) · [生成器](generate.py) · [独立核验](verify_data.py)

ALISTA、HLISTA 和 EBT-LISTA 的原始稀疏编码入口按每条干净观测的中心化标准差设置噪声。NA-ALISTA 的原始实现调用 `torch.std(y, dim=1)`。五个方法共用上述固定验证、测试观测。EBT-LISTA、HLISTA、ALISTA、ELISTA 也用同一套固定训练数据。NA-ALISTA 训练按作者协议在线抽样，细节见下文。不同论文里的同名 SNR 不能直接当成本实验的恢复成绩。[ALISTA 噪声](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L79-L99) · [HLISTA 噪声](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/utils/train.py#L82-L103) · [EBT-LISTA 噪声](https://github.com/lzalza/EBT-LISTA/blob/efe0a317578f47b2ea8defed0fc04cc0044f03f3/code/utils/train.py#L135-L157) · [NA-ALISTA 噪声](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/noise.py#L10-L22)

## 基线

| PyTorch 名称 | 实验所用变体 | 作者实现 |
|---|---|---|
| EBT-LISTA | EBT-LISTA-CPSS，逐层独立更新矩阵。每条样本用变换后残差的二范数除以 50 调整阈值，保留可学习偏置。支持选择每层增加 1.2%，上限 13% | [LISTA_cpss.inference](https://github.com/lzalza/EBT-LISTA/blob/efe0a317578f47b2ea8defed0fc04cc0044f03f3/code/models/LISTA_cpss.py#L114-L160) |
| HLISTA | Hybrid-LISTA-CPSS，采用作者支持的 untied/D 变体。每层独立的两次稀疏更新与三层卷积。支持选择每层增加 0.7%，上限 13% | [作者示例](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/README.md#L113-L121) · [前向计算](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/models/Hybrid_LISTA_cpss.py#L167-L210) |
| ALISTA | 解析矩阵 $W$，逐层学习步长和阈值。支持选择每层增加 1.2%，上限 13% | [ALISTA.inference](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/models/ALISTA.py#L86-L106) |
| NA-ALISTA | `NA_ALISTA_UR_128`，用两个残差特征及跨层共享的 128 维 LSTM 预测每条样本的步长和阈值。支持选择沿用作者的整数序列 | [模型](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/algorithms.py#L125-L174) · [参数序列](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/run.py#L44-L56) |
| ELISTA | 论文主模型，共享可学习矩阵、两次外梯度更新及多阶段阈值 | [论文公式（8）](https://doi.org/10.1609/aaai.v35i10.17032) · [Hybrid ISTA 的参考实现](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/models/ELISTA.py#L149-L158) |

ALISTA 与 NA-ALISTA 使用从当前 $A$ 求出的 $W$。设 $V=(AA^\top)^{-1}A$，第 $j$ 列取 $w_j=v_j/(a_j^\top v_j)$。在 $A$ 满行秩时，这是约束 $\operatorname{diag}(W^\top A)=1$ 下最小化 $\|W^\top A\|_F^2$ 的解。实现用线性方程求解，以 float64 计算 $W$，再保存为 float32。[ALISTA 原目标](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/matlabs/CalculateW.m#L1-L6) · [NA-ALISTA 原目标](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/optimize_matrices.py#L44-L70)

HLISTA 有一处影响批处理的移植修正。作者将卷积输出从 `[batch,n,1]` 直接 reshape 为 `[n,batch]`，在批量大于 1 时会重新排列不同样本的元素。本实验沿信号维做卷积并显式保留样本的行索引。测试比较同一条观测单独推理和组批推理的输出。NA-ALISTA 在相同权重与输入下，与作者 PyTorch 实现逐数值比对三层前向、反传梯度和一次 Adam 更新后的输出。四个有 TensorFlow 参考实现的模型均有 PyTorch 测试，尚无 TensorFlow 与 PyTorch 全流程逐数值比对。[HLISTA 对应原行](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/models/Hybrid_LISTA_cpss.py#L187-L206) · [本地模型](models.py) · [测试](test_experiment.py)

## 五档独立训练

每次训练只接收一个 SNR。四档有限噪声沿用 `data/main`。无噪声数据从同一份已保存的 $A,W,x$ 派生，保存 $y=xA^\top$ 的 float32 结果。训练、验证、测试划分及样本顺序与四个有限档配对。

| 方法与模式 | 参数更新 | 正式预算 |
|---|---|---|
| EBT-LISTA、HLISTA、ALISTA、ELISTA 逐层 | 进入每个深度时先训练新增参数，再对第 1 层至当前层的全部可训练参数联合微调两次。三个子阶段的 Adam 学习率依次为 $5\times10^{-4}$、$10^{-4}$、$10^{-5}$ | 每层三个子阶段**合计最多 1,000 轮**，等分上限为 334、333、333 轮。每个子阶段分别使用 50 轮早停 |
| 上述四个方法端到端 | 从第 16 层的恢复误差向整个模型反传，Adam 学习率 $10^{-3}$ | 最多 100,000 轮，连续 50 轮无改善即停 |
| NA-ALISTA 端到端 | 同一个 128 维 LSTM 展开 16 步，全程反传。Adam 学习率 $2\times10^{-4}$、batch 512、梯度范数上限 1，首个带噪训练批拟合标准化统计 | 默认与本实验一致，最多 100,000 轮，50 轮早停。可单独选择下述作者轮数 |

四个逐层模型的预训练仅更新首次引入的参数。ELISTA 的共享矩阵在第一层引入，之后每次联合微调都重新开放。两次微调均按作者代码在 Adam 更新前将参数梯度乘以 $0.3^{t-j}$，其中 $t$ 为当前深度、$j$ 为参数首次出现的层数。逐层损失为总平方误差的一半，沿用 TensorFlow 的 `l2_loss` 定义。四个端到端模型以均方误差训练，梯度范数上限为 1。固定训练集每档 10,000 条、batch 64，每轮 157 次更新。[ALISTA 分阶段训练](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L201-L237) · [梯度缩放](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L415-L436)

每轮在当前 SNR 的固定验证集上计算所有信号误差总能量与真实信号总能量之比（线性 NMSE）。严格创新低时保存权重并重置等待，持平也累计。开始联合微调时保留同一深度进入该阶段时的最佳权重，并重新计算等待轮数。每个训练运行最终只发布一份完整模型 `best.pt`。训练阶段的指标供检查优化过程使用。[训练配置](training.json) · [训练循环](train.py)

### NA-ALISTA 作者协议与本实验早停

两种设置都按作者的实数合成任务训练。每轮重新生成 50,000 个 Bernoulli–Gaussian 信号，各位置以 $k/n$ 的概率非零，非零幅值服从标准正态。打乱后取 97 个完整的 512 样本批次。对每条 $Ax$，按其样本标准差乘 $10^{-r/20}$ 生成新高斯噪声。无噪声档直接使用 $Ax$。这个训练分布和固定训练集的“恰好 50 个非零、每档噪声方差固定”有差别。各方法的公共验证图与最终测试仍使用相同的存储观测。[作者采样](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/get_data.py#L54-L77) · [作者噪声](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/noise.py#L14-L22)

默认 `--na-schedule experiment` 用固定公共验证集选权重和执行 50 轮早停。`--na-schedule author` 按 $100+20T$ 轮训练，16 层共 420 轮、40,740 次更新。每轮在单独生成的 10,000 条固定干净信号上重新加噪，取完整批次按作者指标选权重，跑满预算。这个内部留出集与项目的测试集独立。两个设置都使用本实验提供的 $A,W$。作者脚本列出的 20 和 40 dB 已扩展为本实验的五档。[作者训练入口](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/run.py#L39-L76) · [训练及选权](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/train_utils.py#L18-L72)

### ELISTA 主模型

按 Li 等（AAAI 2021）公式（8），令 $B\in\mathbb R^{n\times m}$ 为所有层及两个子步共享的可学习矩阵，$x_0=0$。第 $t$ 层有两个步长 $\alpha_{t,1},\alpha_{t,2}$ 和两对阈值 $(a_{t,1},b_{t,1}),(a_{t,2},b_{t,2})$。先求中间估计 $u_t$，再用 $u_t$ 处的观测残差修正本层输入 $x_t$。[论文公式（8）](https://doi.org/10.1609/aaai.v35i10.17032) · [公开实现的两次更新](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/models/ELISTA.py#L149-L158)

$$
\begin{aligned}
u_t &= \operatorname{MT}\big(x_t+\alpha_{t,1}B(y-Ax_t),a_{t,1},b_{t,1}\big),\\
x_{t+1} &= \operatorname{MT}\big(x_t+\alpha_{t,2}B(y-Au_t),a_{t,2},b_{t,2}\big).
\end{aligned}
$$

公式（6）的多阶段阈值 MT 逐坐标作用。在 $0\le a<b$ 时，幅值小于等于 $a$ 的分量置零，大于等于 $b$ 的保留原值，中间区间线性连接。公开 TensorFlow 参考来自 Hybrid ISTA 作者团队，尚未核实 Li 等本人发布的仓库。本实现依其初始化取 $L=1.001\|A\|_2^2$、$B=A^\top/L$、两步长为 1、两对原始阈值均为 $(0.4/L-0.1,0.4/L)$。代码按行存放样本，保存的是 $B^\top$。优化时对阈值的绝对值排序，并在二者相等时使用硬阈值，避免除零。独立 NumPy 测试按两条公式核对每层输出。[矩阵与阈值初始化](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/models/ELISTA.py#L28-L120) · [参考阈值函数](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/utils/tf.py#L126-L136)

Li 等的补充材料给出逐层训练的 batch 64、Adam、基础学习率 $5\times10^{-4}$。验证损失连续 4,000 次迭代未下降时调低学习率，学习率降至 $10^{-5}$ 后若仍无改善则结束该层。本实验沿用逐层训练和学习率范围，按上述每阶段 50 个 epoch 的规则选权重。全深度端到端是本项目增加的对照条件。[作者补充材料 S1](https://zhouchenlin.github.io/Publications/2021-AAAI-ELISTA_supp.pdf)

### 作者代码和旧仓库怎样计数

| 来源 | 训练预算的实际含义 |
|---|---|
| 本仓库初始化前的 PyTorch 版 | `commands.sh` 中的 ALISTA 稀疏恢复命令继承 `--epochs=100000`。16 层，每层依次冻结旧层、以两档更小的学习率微调当前前缀。100,000 是这 48 个子阶段中**每个子阶段**的 epoch 上限。按默认设置每个 epoch 有 51,200 ÷ 128 = 400 次更新。每轮验证一次，连续 5 轮没有改善就结束当前阶段。旧提交未附这次 ALISTA 训练的逐轮日志。 [启动命令](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/commands.sh#L63-L65) · [参数](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/train.py#L36-L56) · [训练循环](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/engine.py#L85-L138) · [阶段](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/engine.py#L382-L429) |
| VITA-Group 的 ALISTA | `--maxit=200000` 按每个子阶段的小批更新计数。循环从 0 到 `maxit`，最多执行 200,001 次更新。每层先训练新变量，再对当前前缀进行两档微调。每 10 步验证，每 100 步检查是否超过等待期。通用等待值为 4,000 步，README 的 16 层 ALISTA 命令设为 2,000 步。训练批次在线生成。 [配置](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/config.py#L163-L176) · [16 层示例](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/README.md#L135-L149) · [阶段](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L189-L237) · [停止判据](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L475-L508) |
| EBT-LISTA 与 HLISTA | 作者稀疏恢复入口也有 `maxit=200000` 的单阶段更新上限、每 10 步验证和默认 4,000 步等待期。 [EBT-LISTA 配置](https://github.com/lzalza/EBT-LISTA/blob/efe0a317578f47b2ea8defed0fc04cc0044f03f3/code/config.py#L197-L210) · [HLISTA 配置](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/config.py#L164-L177) |
| NA-ALISTA | 作者固定 16 层时用 `100 + 20 × 16 = 420` 个 epoch。每轮生成一组训练信号，50,000 条、batch 512、丢弃尾批，每轮 97 次更新，共 40,740 次。20 和 40 dB 分别训练，两个公开日志各有 420 轮记录。 [训练脚本](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/run.py#L44-L55) · [数据加载](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/get_data.py#L65-L77) · [20 dB 日志](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/res/models/model=NA_ALISTA_UR_128__m=250__s=50__k=16__noise=GaussianNoise20__n=500/train_log#L419-L421) · [40 dB 日志](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/res/models/model=NA_ALISTA_UR_128__m=250__s=50__k=16__noise=GaussianNoise40__n=500/train_log#L419-L421) |

上一轮服务器的单档结果记在 [实验报告](experiment_report.md)。该报告对应先前的贪心逐层训练与 5 轮早停。 [早期混合数据首层检查](results/first-layer-budget.csv) 仅用于追溯旧协议。

## 服务器运行

Python 3.12，PyTorch 2.8，NumPy 2.3.5。已有 `data/main` 的服务器可直接运行网格。首次使用时先生成有限四档数据。网格会从原数据派生缺少的无噪声数据。输出默认写到新目录 `runs/single-snr-finetuned`。

```bash
python -m pip install -r requirements.txt
# 仅在首次创建数据时执行
python generate.py --config experiment.json --out data/main
python verify_data.py data/main
python run_single_snr_grid.py
```

单次运行可用以下命令，支持 `--snr 40|20|10|5|inf`。在同一输出目录重跑已完整训练的同配置任务会复用检查点。修改了实验配置时使用新的输出目录。

```bash
python train.py train --model ELISTA --mode layerwise --snr 5
python train.py train --model ELISTA --mode end_to_end --snr 5
python train.py test --model ELISTA --mode layerwise --snr 5
python train.py train --model NA-ALISTA --mode end_to_end --snr 20 --na-schedule author --out runs/na-author
```

独立生成无噪声数据可运行 `python generate.py --noiseless-from data/main --out data/noiseless`。GPU 网格遵守 `CUDA_VISIBLE_DEVICES`。单机 CPU 可加 `--device cpu`。小规模检查可使用 [快速配置](training.quick.json)。

## 从完整模型绘制深度曲线

每个“模型、训练方式、SNR、种子”加载一次最终 `best.pt`。对同一批输入单次展开完整网络，读取全部 16 个输出。每个深度在完整数据划分上累加能量，输出 $10\log_{10}(\sum_i\|\hat x_i-x_i\|_2^2/\sum_i\|x_i\|_2^2)$。表中记录每条曲线的 checkpoint SHA-256。图中的不同点共享同一组参数，NA-ALISTA 也共享同一次展开的 LSTM 状态。逐层联合微调模式和端到端模式分行展示。

```bash
python plot_single_snr.py --runs runs/single-snr-finetuned
python plot_single_snr.py --runs runs/single-snr-finetuned --split test
```

完整网格结束后会自动绘制验证图。数据在 `fixed_depth_val_nmse.csv`，图在 `figures/fixed_depth_val_nmse.png` 和同名 PDF。端到端逐样本误差箱线图也由最终权重计算。精确恢复的样本在逐样本表中保留为 $-\infty$ dB，箱线图和四分位数使用有限 dB 样本，并单列精确恢复数。在只有上一轮四模型权重的服务器，可用 `--runs runs/single-snr --methods EBT-LISTA HLISTA ALISTA NA-ALISTA --include-legacy-na` 读取历史模型。旧结果的图应按当时的训练方式解读。

作者源码版本与来源列在 [upstreams.json](upstreams.json)。可用 `python prepare.py <模型>` 查看作者或参考实现及实验数据适配器。使用适配器时设置 `RECOVERY_DATA` 和单一 `RECOVERY_SNR`。本项目的正式训练和绘图分别由 [train.py](train.py) 与 [plot_single_snr.py](plot_single_snr.py) 执行。
