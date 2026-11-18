# 高噪声观测下的稀疏信号恢复

给定已知字典 $A$ 和带噪观测 $y$，估计原始稀疏信号 $x$。首版固定字典，比较每个方法在 40、20、10、5 dB 下的恢复误差。40 dB 用作低噪声参照，10 和 5 dB 是主要比较条件。数据用 NumPy 生成一次，训练和评估统一使用 PyTorch。正式比较包括四个基线、两种训练方式，每组训练一个可跨四档 SNR 使用的模型。

## 数据协议

| 项目 | 首版值 |
|---|---|
| 字典 | $A\in\mathbb R^{250\times500}$，独立标准高斯元素，逐列归一化至单位二范数。三个数据划分共用同一张字典 |
| 稀疏信号 | 每个样本从 500 个位置均匀无放回抽取 50 个位置。非零系数独立服从标准正态分布，保留自然幅值 |
| 监督标签 | 无噪声的稀疏系数 $x$ |
| 数据量 | 训练 10,000、验证 1,000、测试 2,000 条独立信号。每条信号生成四个观测 |
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

ALISTA、HLISTA 和 EBT-LISTA 的原始稀疏编码入口按每条干净观测的中心化标准差设置噪声。NA-ALISTA 的原始实现调用 `torch.std(y, dim=1)`。本实验对所有基线采用上述统一功率口径。不同论文里的同名 SNR 不能直接当成本实验的恢复成绩。[ALISTA 噪声](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L79-L99) · [HLISTA 噪声](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/utils/train.py#L82-L103) · [EBT-LISTA 噪声](https://github.com/lzalza/EBT-LISTA/blob/efe0a317578f47b2ea8defed0fc04cc0044f03f3/code/utils/train.py#L135-L157) · [NA-ALISTA 噪声](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/noise.py#L10-L22)

## 基线

| PyTorch 名称 | 实验所用变体 | 作者实现 |
|---|---|---|
| EBT-LISTA | EBT-LISTA-CPSS，逐层独立更新矩阵。每条样本用变换后残差的二范数除以 50 调整阈值，保留可学习偏置。支持选择每层增加 1.2%，上限 13% | [LISTA_cpss.inference](https://github.com/lzalza/EBT-LISTA/blob/efe0a317578f47b2ea8defed0fc04cc0044f03f3/code/models/LISTA_cpss.py#L114-L160) |
| HLISTA | Hybrid-LISTA-CPSS，采用作者支持的 untied/D 变体。每层独立的两次稀疏更新与三层卷积。支持选择每层增加 0.7%，上限 13% | [作者示例](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/README.md#L113-L121) · [前向计算](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/models/Hybrid_LISTA_cpss.py#L167-L210) |
| ALISTA | 解析矩阵 $W$，逐层学习步长和阈值。支持选择每层增加 1.2%，上限 13% | [ALISTA.inference](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/models/ALISTA.py#L86-L106) |
| NA-ALISTA | `NA_ALISTA_UR_128`，用两个残差特征及跨层共享的 128 维 LSTM 预测每条样本的步长和阈值。支持选择沿用作者的整数序列 | [模型](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/algorithms.py#L125-L174) · [参数序列](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/run.py#L44-L56) |

ALISTA 与 NA-ALISTA 使用从当前 $A$ 求出的 $W$。设 $V=(AA^\top)^{-1}A$，第 $j$ 列取 $w_j=v_j/(a_j^\top v_j)$。在 $A$ 满行秩时，这是约束 $\operatorname{diag}(W^\top A)=1$ 下最小化 $\|W^\top A\|_F^2$ 的解。实现用线性方程求解，以 float64 计算 $W$，再保存为 float32。[ALISTA 原目标](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/matlabs/CalculateW.m#L1-L6) · [NA-ALISTA 原目标](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/optimize_matrices.py#L44-L70)

HLISTA 有一处影响批处理的移植修正。作者将卷积输出从 `[batch,n,1]` 直接 reshape 为 `[n,batch]`，在批量大于 1 时会重新排列不同样本的元素。本实验沿信号维做卷积并显式保留样本的行索引。测试会比较同一条观测单独推理和组批推理的输出。NA-ALISTA 在相同权重与输入下，已用作者 PyTorch 实现逐数值比对三层前向。三个 TensorFlow 来源的模型已做 PyTorch 路径测试，尚无跨框架逐数值比对。[HLISTA 对应原行](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/models/Hybrid_LISTA_cpss.py#L187-L206) · [本地模型](models.py) · [测试](test_experiment.py)

## 两种训练方式

正式配置采用 16 层、batch size 64、Adam 学习率 $10^{-3}$，梯度范数上限 1。每个 epoch 遍历 40,000 条带噪训练观测并更新参数 625 次。**每个训练阶段最多 100,000 个 epoch，每轮结束验证一次。连续 5 个 epoch 没有改善，立即结束当前阶段。** [正式配置](training.json) · [训练入口](train.py)

| 方式 | 训练目标和参数更新 |
|---|---|
| `end_to_end` | 全部 16 层联合训练，整网作为一个阶段。以最终层预测和真实信号的均方误差反传，选用该阶段验证指标最低的权重 |
| `layerwise` | 深度由 1 增至 16，每层是一个独立阶段，分别执行上限和早停。ALISTA、EBT-LISTA、HLISTA 冻结已有层，只更新新增层。每一层恢复自己的最佳权重，再进入下一层 |
| NA-ALISTA 的 `layerwise` | 维持同一个共享 LSTM，前 $t-1$ 步停止梯度，从第 $t$ 步更新控制器。共享权重发生变化后，之前各步在下次前向时也会使用新权重。初始隐状态、细胞状态只在首阶段更新 |

两种方式都在混合 SNR 上训练，按四档验证集**线性 NMSE 的等权平均**选权重，测试集单独评估。每档报告信号总误差能量与真实信号总能量之比，以及其 dB 值。另报告 top-$k$ 支撑 F1，取预测幅值最大的 $k$ 个坐标与真实支撑求交集。幅值相同时，坐标索引小者优先。这里真实与预测的支撑大小都为 $k$，因此 F1 等于命中数除以 $k$。NA-ALISTA 的残差标准化只用首个训练批次拟合，均值和标准差随权重保存。

改善指当前四档平均线性验证 NMSE 严格低于本阶段历史最低值。持平或变差均累计一轮，出现新低就将计数归零。第一轮建立最佳值，因此最快在第 6 轮结束。达到上限也会恢复最佳权重。端到端只有一个计数器，逐层训练每增一层就重新计数。各模型实际用时与更新次数由验证曲线决定，日志会给出实际值。需要固定预算的快速检查时可选 [32 轮配置](training.quick.json)。

### 作者代码和旧仓库怎样计数

| 来源 | 训练预算的实际含义 |
|---|---|
| 本仓库初始化前的 PyTorch 版 | `commands.sh` 中的 ALISTA 稀疏恢复命令继承 `--epochs=100000`。16 层，每层依次冻结旧层、以两档更小的学习率微调当前前缀。100,000 是这 48 个子阶段中**每个子阶段**的 epoch 上限。按默认设置每个 epoch 有 51,200 ÷ 128 = 400 次更新。每轮验证一次，连续 5 轮没有改善就结束当前阶段。旧提交未附这次 ALISTA 训练的逐轮日志。 [启动命令](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/commands.sh#L63-L65) · [参数](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/train.py#L36-L56) · [训练循环](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/engine.py#L85-L138) · [阶段](https://github.com/Micuez/ISTA-class/blob/3b9bf66c65458a67e69f72f7cb3d3382fb29c165/engine.py#L382-L429) |
| VITA-Group 的 ALISTA | `--maxit=200000` 按每个子阶段的小批更新计数。循环从 0 到 `maxit`，最多执行 200,001 次更新。每层先训练新变量，再对当前前缀进行两档微调。每 10 步验证，每 100 步检查是否超过等待期。通用等待值为 4,000 步，README 的 16 层 ALISTA 命令设为 2,000 步。训练批次在线生成。 [配置](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/config.py#L163-L176) · [16 层示例](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/README.md#L135-L149) · [阶段](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L189-L237) · [停止判据](https://github.com/VITA-Group/ALISTA/blob/23f3637f01066a919d37af76fd60ee03d33188b9/utils/train.py#L475-L508) |
| EBT-LISTA 与 HLISTA | 作者稀疏恢复入口也有 `maxit=200000` 的单阶段更新上限、每 10 步验证和默认 4,000 步等待期。 [EBT-LISTA 配置](https://github.com/lzalza/EBT-LISTA/blob/efe0a317578f47b2ea8defed0fc04cc0044f03f3/code/config.py#L197-L210) · [HLISTA 配置](https://github.com/ZhengZY-EE/Hybrid_ISTA/blob/ddde3c5867bfb453741d0c2e2cca507e6070994a/config.py#L164-L177) |
| NA-ALISTA | 作者固定 16 层时用 `100 + 20 × 16 = 420` 个 epoch。每轮生成一组训练信号，50,000 条、batch 512、丢弃尾批，每轮 97 次更新，共 40,740 次。20 和 40 dB 分别训练，两个公开日志各有 420 轮记录。 [训练脚本](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/run.py#L44-L55) · [数据加载](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/get_data.py#L65-L77) · [20 dB 日志](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/res/models/model=NA_ALISTA_UR_128__m=250__s=50__k=16__noise=GaussianNoise20__n=500/train_log#L419-L421) · [40 dB 日志](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/res/models/model=NA_ALISTA_UR_128__m=250__s=50__k=16__noise=GaussianNoise40__n=500/train_log#L419-L421) |

以 VITA 的 16 层 README 示例为参照，从头跑完 48 个子阶段，即使每个阶段的最佳验证值都在第 0 步，也要到第 2,100 步才会触发早停，每阶段至少执行 2,101 次更新，合计 100,848 次。实际完成步数需由训练日志确认。本实验沿用旧 PyTorch 仓库的 epoch 上限和早停口径。逐层模式按本实验上表的定义，每层训练一次新增层或共享控制器。比较训练投入时应读取每层实际执行的轮数和更新次数。

**首层验证。** 在正式的混合 SNR 数据上，固定权重种子 0 和 batch 64，从头训练第一层，比对两种方法的前 8 轮。分别在前 2 轮和前 8 轮内按四档平均线性验证 NMSE 选权重。ALISTA 分别选中第 2 和第 6 轮，5 dB 验证 NMSE 从 −3.072 改善至 −3.126 dB。EBT-LISTA 分别选中第 2 和第 8 轮，四档平均线性 NMSE 从 0.342727 降至 0.341971，但 5 dB 从 −3.485 升至 −3.345 dB。用正式早停规则单独检查 ALISTA 首层时，第 6 轮取得最优值，第 7–11 轮均未改善，于第 11 轮结束并恢复第 6 轮权重，共 6,875 次更新。整组检查覆盖单种子的首层。复查时复制 `training.quick.json`，将 `depth` 设为 1、`total_epochs` 设为 8，分别对两个模型运行 `layerwise`，比较 `history.jsonl` 中的验证记录。复查 ALISTA 的早停时，复制正式配置并仅将 `depth` 设为 1。[逐轮、逐档结果](results/first-layer-budget.csv)

## 运行

数据生成与核验可用 Python 3.12 和 NumPy 2.3.5 单独运行。训练环境已用 Python 3.12、PyTorch 2.8.0 CPU 版验证。下面的命令在项目根目录执行，`python` 指已选环境的解释器。本机独立环境的解释器为 `.venv\Scripts\python.exe`。

```sh
python -m pip install -r requirements.txt
python generate.py --config experiment.json --out data/main
python verify_data.py data/main
python -m unittest -v test_experiment
```

新克隆的工作目录先生成 `data/main`。已有这份数据的工作目录可直接核验。修改数据配置后用新的输出目录，例如 `data/v2`。字典、解析矩阵、各划分的 `x.npy` 和四档 `y_*dB.npy` 都有 SHA-256 记录，见对应的 `manifest.json`。Git 管理配置、源码、测试与补丁；生成的数据和模型权重保存在本地。

```sh
python train.py train --model ALISTA --mode end_to_end --out runs/early-stop
python train.py train --model ALISTA --mode layerwise --out runs/early-stop
python train.py train --model all --mode both --out runs/early-stop
python train.py test  --model all --mode both --out runs/early-stop
```

可用 `--data`、`--config`、`--seed` 和 `--device cpu` 或 `--device cuda` 指定数据、训练配置、权重种子与设备。使用上一版的 32 轮预算时，传入 `--config training.quick.json --out runs/quick-32`。上面的正式配置命令将结果写入 `runs/early-stop/<模型>/<训练方式>/seed-<种子>/`。其中 `history.jsonl` 给出每轮分档验证结果，`run.json` 记录每阶段的实际轮数、更新次数、最佳轮次、停止原因与可更新参数，`best.pt` 保存权重，`test.json` 由独立测试命令写入。改动超参数或实现后使用新的 `--out` 目录保留实验版本。

## 作者仓库的数据入口

[版本清单](upstreams.json) 固定了四个作者仓库的提交。[补丁目录](patches/) 保存了针对这些版本的数据入口改动。[共享适配器](experiment_inputs.py) 向旧 TensorFlow 入口提供列批次，向 PyTorch 提供行批次。如需查阅应用补丁的作者代码，可运行 `python prepare.py ALISTA`，其他方法用表中的名称，也可通过 `--from-checkout` 指定含锁定提交的本地 Git 仓库。统一 PyTorch 训练入口可独立运行。

在作者原框架中使用补丁时，将本项目根目录加入 `PYTHONPATH`，用 `RECOVERY_DATA` 指向生成的数据目录。三个 TensorFlow 入口的 `run_sc_train` 读取共享训练及验证观测，`run_sc_test` 按 `RECOVERY_TEST_DB` 读取指定测试观测，并按档位保存结果。NA-ALISTA 的 `run.py` 读取本实验的 $m,n,k$，`train_model` 使用已加噪的 $y$ 和独立验证集。这些入口保留作者原框架依赖，双训练方式的正式比较统一走本项目的 `train.py`。
