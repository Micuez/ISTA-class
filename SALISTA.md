# SALISTA V3 设计与验证记录

**共享 GRU 已能在主字典、真实稀疏度 50 上产生非零输出和梯度。** 短配置为即将被全部压零的样本启用阈值缩放。四档 SNR 每档 10,000 条训练观测的首层均有非零输出。严格证书是单独的对照，设 `stall_threshold_ratio=null` 即可复现。本文记录 CPU 诊断与短训练验证，正式恢复成绩仍待实验。

## 项目中可以复用什么

| 对象 | 当前仓库的实现与训练约定 |
|---|---|
| 字典与解析权重 | `generate.analytic_weights` 用双精度求解 `(AAᵀ)V=A`，再将每列除以 `aᵢᵀvᵢ`。满行秩、列非零时得到 ALISTA 的 Frobenius 目标解。常规训练读取同一份 `A.npy` 和 `W.npy`。直接构造 `SALISTA(A, None, s, depth, ...)` 时，在 CPU 离线计算一次 `W`。两条路径都将 `A,W` 注册为固定缓冲区。 |
| NA-ALISTA | 一个 128 维 LSTMCell 和回归头跨全部展开步共享。首个带噪训练批校准两个残差的一范数，初始隐状态和记忆状态可训练。从记忆状态预测每个样本的两个标量。基线网格使用端到端。六法正式配置另开放共享 LSTM 的递增深度对照。 |
| NA-ALISTA 的训练数据 | 每轮重新生成 Bernoulli–Gaussian 信号，非零个数随机。噪声按各条干净观测的标准差缩放。默认 batch 512、Adam 学习率 0.0002、MSE、梯度范数上限 1。`na_schedule=experiment` 用公共固定验证集早停，`author` 跑满 `100+20K` 轮并用独立合成留出集选权重。 |
| ALISTA、EBT-LISTA、HLISTA、ELISTA | 逐层模式先训练新参数，再对当前全部层作两档低学习率微调。此路径最小化平方误差总和的一半，联合阶段缩放旧层梯度。ELISTA 的学习矩阵跨层共享。它们也支持全深度端到端。 |
| 旧实验报告 | `experiment_report.md` 中的 NA-ALISTA 逐层结果属于历史截断梯度协议。它有不同的数据、早停和更新预算，无法据此估计本次 SALISTA 微调的收益。 |

实现依据见[本地模型](models.py)、[训练器](train.py)、[NA 在线采样](na_training.py)、[数据生成器](generate.py)、[NA-ALISTA 论文（ICLR 2021）](https://arxiv.org/abs/2010.01930)和[作者模型代码](https://github.com/feeds/na-alista/blob/40597f05e3ef179e4ae44173d25674640db7d27a/utils/algorithms.py#L125-L174)。

## 一层计算与参数归属

对每个样本，观测为 $y=Ax^\star+\xi\in\mathbb R^m$。目标 $x^\star\in\mathbb R^n$ 的支撑 $S$ 满足 $|S|\le s$。$A,W\in\mathbb R^{m\times n}$ 固定。令 $x^{-1}=x^0=0$，在第 $k$ 层计算 $r=y-Ax^k$、$c=W^\top r$、$v=x^k-x^{k-1}$。实现将批内样本按行存储。

$$
z=x^k+\gamma\odot c+\beta\odot v,\qquad
x^{k+1}=\operatorname{sign}(z)\odot\max(|z|-\theta,0).
$$

| 对象 | 来源 |
|---|---|
| 全局历史状态 | 每样本一份零初态。将 $(\|r\|_1,\|c\|_1,\|c\|_\infty,\|v\|_1,B_k)$ 逐项取 `log1p` 后输入 GRU 或 LSTM。LSTM 用隐状态输出，记忆状态也传到下一步。 |
| 坐标表示 | 将全局隐状态的线性映射与局部特征 $(x_i,c_i,v_i,\log(1+E_i),q_i,\omega_i)$ 的线性映射相加并取 `tanh`。编码权重和三个输出头在坐标间共享。特征缩放无需拟合统计量。 |
| 步长与动量 | 两个输出头分别通过 sigmoid 映射到 $[\gamma_{\min},\gamma_{\max}]$ 和 $[0,\beta_{\max}]$。第一步令 $\beta=0$。动量逐坐标相乘。 |
| 阈值 | 步长给出解析下界 $L_i=\gamma_i(\Psi_i+\varepsilon\omega_i)$。阈值头额外接收 $\log(1+L_i)$，以 softplus 提议增量。逐样本作欧氏投影到 $\{\delta\ge0:\sum_i\delta_i\le\eta B_k\}$，得到参考阈值 $\theta^{\rm ref}=L+\delta$。严格模式直接使用它。短配置按下述条件计算实际阈值。 |
| 参数更新 | 优化器更新循环单元、特征融合层和输出头。三个控制向量随样本和层动态生成。字典、解析权重、证书以及步长区间、动量上限和预算系数由输入或配置给定。`beta_max=0` 可消去动量，`gamma_min=gamma_max` 可固定步长，`eta=0` 可消去阈值增量。 |

示例选宽度 32、步长区间 $[0.5,1.1]$、$\beta_{\max}=0.2$、$\eta=0.05$。这些是原型超参数。它们对任意字典的收敛或加速均有待另证。

## 证书的条件、计算和保证边界

输入需给出 $B_0\ge\|x^\star\|_1$ 与 $\varepsilon\ge\|\xi\|_2$。令 $C=W^\top A$，$d_i=C_{ii}$，$D$ 为 $|C|$ 将对角线置零后的矩阵，$\mu=\max_{ij}D_{ij}$，$\omega_i=\|W_{:,i}\|_2$。记 $T_j(u)$ 为非负向量 $u$ 最大的至多 $j$ 项之和，$T_0=0$。设 $q_i=T_{s-1}(D_{i,:})$ 和 $g=\min_i(|d_i|-q_i)$。实现检查对角接近 1，并在公式中使用实际的 $d_i$。

若当前两步的支撑均包含于 $S$，令 $e=x^k-x^\star$。由 $c=-Ce+W^\top\xi$，在 $S$ 上逐行取绝对值得 $|d_i||e_i|\le |c_i|+\varepsilon\omega_i+q_i\|e\|_\infty$。在误差最大的坐标上使用此式，得 $g>0$ 时的上界 $H=(\|c\|_\infty+\varepsilon\max_i\omega_i)/g$。据此计算

$$
E_i=\begin{cases}
\min(B_k,(|c_i|+\varepsilon\omega_i+q_i H)/|d_i|), &g>0,\\
B_k, &g\le0,
\end{cases}
\qquad
\Psi_i=\min\{T_s(D_{i,:}\odot E),\mu B_k\}.
$$

在 $S$ 外，已有两步的值与对角动量均为零，且 $|c_i|\le\Psi_i+\varepsilon\omega_i$。因此 $|z_i|\le L_i\le\theta_i^{\rm ref}$。从两个零初态归纳可得，全程使用参考阈值的样本在输入界成立时无虚假激活。真实支撑内的小系数也可能被压成零。

设 $B_{-1}=B_0$，$G=\max_i\gamma_i$，$\omega_*=\max_i\omega_i$。在 $S$ 内，用 $|\mathcal S_\theta(z)-z|\le\theta$ 估计收缩偏差，对各项求和。线性误差项使用**列和**，阈值总量使用 $\sum_{i\in S}\theta_i\le sG(\mu B_k+\varepsilon\omega_*)+\eta B_k$。得到

$$
a_0=\max_j\bigl[|1-\gamma_j d_j|+T_{s-1}(\gamma\odot D_{:,j})\bigr],\qquad
R=(a_0+sG\mu+\eta+\beta_{\max})B_k+\beta_{\max}B_{k-1}
       +2sG\omega_*\varepsilon .
$$

实现对原讨论的递推做了一处收紧，取 $B_{k+1}=\min(R,B_0+\|x^{k+1}\|_1)$。第二项直接来自三角不等式，能抑制预算在保守字典上膨胀。行、列方向分别用非对称 $W^\top A$ 和独立 NumPy 计算测试。[证书实现](salista_constraints.py) · [数值测试](test_salista.py)

上述性质针对精确算术及合法输入上界。若无噪且 $a_0+sG\mu+\eta\le\bar a$ 对每层一致成立，$\bar a+2\beta_{\max}<1$ 是该二阶上界递推几何衰减的充分条件。有限精度诊断使用数值容差。精确支撑恢复、任意学习动量的加速率，以及固定正则系数的 LASSO 最优性均缺少这里所需的额外前提。证书存储为 $O(n^2)$，样本相关的行、列加权 top-$s$ 最坏需访问 $O(n^2)$ 个元素，另计选择操作。代码分块以限制单个工作张量，反向传播仍需保留中间量。

### 显式界与高斯概率界

`salista.bounds` 可取 `{"mode":"explicit","initial_l1_bound":2.0,"noise_l2_bound":0.02}`。这两个值应来自已知信号和噪声约束。固定训练、验证或测试样本的真实误差只用于事后核验。

`{"mode":"gaussian","failure_probability":0.01}` 对应项目 `generate.py` 的恰好 $s$ 个标准高斯系数和每维标准差 $\sigma_r=\sqrt{s/m}\,10^{-r/20}$ 的高斯噪声。令 $p\in(0,1)$ 为**一个新样本**的总失败概率上限，$t=\log(2/p)$，$R_d=\sqrt{d+2\sqrt{dt}+2t}$。取

$$
B_0=\sqrt{s}\,R_s,\qquad \varepsilon=\sigma_r R_m .
$$

标准高斯向量 $u\in\mathbb R^d$ 满足 $\Pr(\|u\|_2>R_d)\le e^{-t}$。将其用于信号和噪声，结合 $\|x^\star\|_1\le\sqrt{s}\|x^\star\|_2$ 与并集界，两条输入界同时成立的概率至少为 $1-p$。同一事件覆盖整条展开路径。若目标是对 $N$ 个信号同时覆盖，可将每样本失败预算设为总体预算除以 $N$。无噪档取 $\sigma_r=0$。NA 的 Bernoulli 支撑和按样本缩放的噪声需要单独建模。[高斯二次型尾界，命题 1，取单位矩阵](https://arxiv.org/pdf/1110.2842)

### 六法在线对比使用的概率界

`training.comparison.json` 使用 `online_gaussian`。在 NA 作者采样中 $K\sim\mathrm{Binom}(n,s/n)$ 是实际非零数，$s$ 是均值。给定 $p$，令 $t=\log(3/p)$。用独立 Bernoulli 变量的 Bernstein 上尾界，取

$$
k_*=\min\!\left(n,\left\lceil s+\sqrt{2s(1-s/n)t}+\tfrac23t\right\rceil\right).
$$

于是 $\Pr(K>k_*)\le p/3$。给定 $K\le k_*$，用上述卡方尾界取 $\|x^\star\|_2\le R_{k_*}$。沿用 `torch.std` 的样本标准差约定，有 $\mathrm{std}(Ax^\star)\le\|A\|_2 R_{k_*}/\sqrt{m-1}$。对标准高斯噪声向量再用一次卡方尾界，三个事件由并集界同时覆盖。证书的稀疏度参数使用 $k_*$，初始一范数半径取 $\sqrt{k_*}R_{k_*}$，噪声二范数半径取

$$
10^{-r/20}R_m\max\!\left(\sqrt{s/m},\,\|A\|_2 R_{k_*}/\sqrt{m-1}\right).
$$

最大值的第一项覆盖公共验证和测试集的固定方差。主字典在 $p=0.01$ 时得到 $k_*=77$，20 dB 的噪声半径约 3.171。这些是采样前的单样本概率界。缩放阈值仍会中断支撑包含证明。两个循环模型使用同一个在线生成函数并按全局 epoch 配对；作者原轮数与独立留出集仍属于原基线的 `author` 日程。[正式配置](training.comparison.json) · [数据生成](na_training.py)

## 两种训练入口，四种参数共享组合

| 比较条件 | 当前深度的参数与梯度 | 主要代价与解释 |
|---|---|---|
| 共享 GRU / LSTM，端到端 | 深度 $K$ 的最终 MSE 沿所有 $K$ 步回传。一个控制器和三头反复使用。 | 首要基线。深层信号直接决定参数，训练时保留整条展开计算图。共享 LSTM 可独立检验单元类型。 |
| **共享 GRU / LSTM，递增深度** | 深度 $1,\ldots,K$ 都对当前完整前缀反传。到 $K$ 后，再以两档更低学习率微调同一组权重。 | 优先比较的课程训练。每次更新都可能改变较浅层输出。前期反传链短，引入了额外的优化阶段与数据访问。 |
| 独立 GRU / LSTM，端到端 | 每个深度有独立的单元及坐标头，状态沿深度传递，全图联合反传。 | 可检验逐层专门化。固定宽度时参数随深度增加，不能把差异全归因于共享。 |
| 独立 GRU / LSTM，逐层 | 每个深度先冻结旧模块、只训练新模块，再分两档学习率微调所有已启用模块。 | 可以真正冻结旧层。每个子阶段重建 Adam，联合阶段保留进入该阶段时最佳的同深度验证权重。 |

将共享单元仅在最后一次调用上求导会得到截断梯度，更新仍会改变旧层输出。当前共享方案保留完整前缀梯度。全部独立的配置也跨层传递循环状态，第一层没有可训练的动量头。宽度 32、深度 16 时，可训练参数量分别为共享 GRU 5,092、共享 LSTM 6,340、独立 GRU 81,439、独立 LSTM 101,407。

SALISTA 短配置使用单 SNR 固定训练集。六法正式配置中，SALISTA 与 NA-ALISTA 共用作者在线采样。SALISTA 的损失为最终激活深度的元素均方误差（MSE），使用 Adam 与梯度范数上限 1。固定验证集用总误差能量除以总目标能量的线性 NMSE 选权重。示例从 $2\times10^{-4}$ 起步，微调因子为 0.2、0.02，短配置独立层的旧梯度缩放设为 1。正式基线的旧层梯度衰减为 0.3。短配置的每层上限为三轮，共享方案在最后深度将三轮分配给三个子阶段。[训练计划](train.py) · [短配置](training.salista.json)

## 严格模式的首层诊断

在 `data/main` 实测 $m=250,n=500,s=50$，$\mu=0.2113321$，$\max_i q_i=5.274068$，$g=-4.274068$。取示例 $p=0.01$，得到 $B_0=68.24555$。第一步从零开始，任一坐标激活的必要条件是 $|(W^\top y)_i|>\Psi_i+\varepsilon\omega_i$，步长在两侧消去。以下检查逐档扫描全部 10,000 条训练观测，四档共用相同的干净信号。

| SNR | 最大 $\lvert(W^\top y)_i\rvert$ | 最小 $L_i/\gamma_i$ | 至少有一个坐标可能激活的样本数 |
|---|---:|---:|---:|
| 40 dB | 5.067 | 14.529 | 0 / 10,000 |
| 20 dB | 5.162 | 15.485 | 0 / 10,000 |
| 10 dB | 5.390 | 17.781 | 0 / 10,000 |
| 5 dB | 5.650 | 20.395 | 0 / 10,000 |

本配置下 $s\gamma_{\min}\mu>5$。若输出为零，递推候选 $R\ge B_k=B_0$，三角候选也等于 $B_0$。因此预算、残差和干扰界在下一步复现。结合严格的首层间隔可知，这些观测对任意允许的网络权重、任意展开深度都输出零，恢复损失对控制器的梯度为零。在 20 dB 的前 64 条训练观测上另作三层真实前向，逐层非零输出样本数均为 0。这是有限训练数据和当前参数范围的结论。

一个有学习信号的小问题使用同一字典、重新采样的 $s=4$ 信号、20 dB、$p=0.01$ 和 64 条观测。此时 $\max q_i=0.521387$，三步各有 38 条样本输出非零，循环单元及三个输出头均收到非零梯度。64 条的幅值界都满足，其中 1 条噪声范数超过上界（0.23259 对 0.23096），三步均观测到 0 个虚假非零。该批仅用于检查激活与反传。

复核小批使用 NumPy `default_rng(20261118)` 和 `torch.manual_seed(0)`，先顺序生成每条样本的 4 个标准正态值，再以 `choice(500,4,replace=False)` 取位置，最后生成 `(64,250)` 标准正态噪声。信号与噪声转为 float32，取三步最终 MSE 反传一次。主数据清单 SHA-256 为 `0474f621542520e1a2d2714fb346753b99f47c3c72049ae4f421f1ea041724b6`。

### 严格模式的优化器检查

在主数据 20 dB 的前 8 条训练观测上，取严格模式、种子 0，执行一次三层前向、反传与 Adam 更新。MSE 为 0.114046，线性 NMSE 为 1，控制器的最大绝对梯度和权重最大绝对改变量均为 0。

另用本项目 `generate.py` 的独立随机流生成小数据集。保留原字典种子，取 $s=4$、20 dB、训练/验证/测试分别 64/16/16 条。两种日程均使用三层共享 GRU、相同初始化和 9 次更新。配置使用 `stall_threshold_ratio=null`、`max_epochs_end_to_end=9`、`early_stopping_patience=10`。分别重载最佳检查点，通过正式测试入口完成评估。

| 训练日程 | 初始验证线性 NMSE | 选中权重的验证线性 NMSE | 相对初始化的权重最大绝对变化 |
|---|---:|---:|---:|
| 全深度端到端 | 0.778596 | 0.775165 | 0.001820 |
| 递增深度及联合微调 | 0.778596 | 0.775669 | 0.001444 |

这两次运行验证优化器、选权、保存和重载能协同工作。16 条验证信号、单个种子和 9 次更新不足以给两种日程排出性能高低。

## 样本即将全零时的阈值调整

若 $z\ne0$ 且参考阈值会使整条输出为零，取预设比例 $\rho=0.8$，计算

$$
\alpha=\rho\max_{i:\theta_i^{\rm ref}>0}\frac{|z_i|}{\theta_i^{\rm ref}},
\qquad \theta_i^{\rm used}=\alpha\,\theta_i^{\rm ref}.
$$

这样保留了坐标阈值之间的比例。达到最大比值的坐标经收缩后保留 $(1-\rho)|z_i|>0$ 的幅值。只要参考更新已经有非零输出，就取 $\alpha=1$。$z=0$ 时输出自然为零。0.8 是一次预设的折中，尚未调参。该调整属于经验方法。

某条轨迹首次缩放后，`support_certified` 在本次展开余下各步均为假。代码对该样本回到 $B_k=B_0+\|x^k\|_1$ 及 $E_{i,k}=B_k$，不再使用依赖误差支撑至多为 $s$ 的递推。诊断分别返回参考阈值 `reference_theta`、实际阈值 `theta`、缩放比例 `threshold_scale` 及逐样本标记。高斯输入界仍有原有的概率含义，触发缩放后无虚假激活的证明失效。

种子 0、三层共享 GRU。除完整训练集的第一列外，其余各列使用每档存储顺序的前 64 条训练信号，按第三层最终 MSE 反传一次。

| SNR | 首层非零样本 / 全训练集 | 第三层非零样本 | 第三层真阳性 / 假阳性坐标数 | 更新前线性 NMSE |
|---|---:|---:|---:|---:|
| 40 dB | 10,000 / 10,000 | 64 / 64 | 143 / 0 | 0.9475 |
| 20 dB | 10,000 / 10,000 | 64 / 64 | 148 / 0 | 0.9471 |
| 10 dB | 10,000 / 10,000 | 64 / 64 | 160 / 0 | 0.9464 |
| 5 dB | 10,000 / 10,000 | 64 / 64 | 159 / 4 | 0.9482 |

每档这 64 条在三个展开步都触发了缩放。GRU 和步长、阈值、动量头在四档均收到非零梯度，单次 Adam 更新的最大权重改变量约为 0.0002。以 20 dB 为例，真实非零坐标共有 3,200 个，只检出 148 个，恢复精度仍有较大改进空间。

在主数据 20 dB 上又做了短训练检查。种子 0，使用第 0 个 epoch 打乱后的前 8 个训练批（每批 64 条），最终层 MSE、Adam 学习率 0.0002、梯度范数上限 1。固定验证集的前 64 条在更新前后均有 64 条非零输出，线性 NMSE 从 0.944066 降至 0.942972。这个数字只反映一次短训练。

## 使用与后续实验

从项目目录运行以下检查。第一条包含临时目录里的微型训练与检查点回读，第二条并排报告严格阈值的首层激活必要条件和当前配置的实际非零输出数量。

```bash
python -m unittest -v test_salista test_experiment
python check_salista.py --data data/main --config training.salista.json
```

`train.py` 接受 `--model SALISTA` 以及 `--mode end_to_end` 或 `--mode layerwise`，默认选择 SALISTA 的三层短配置。`salista.cell` 取 `gru` 或 `lstm`，`salista.sharing` 取 `shared` 或 `untied`。不同结构或预算应指定不同 `--out`，便于按配置和代码指纹追溯。以下是有学习信号的数据目录上的命令形式。

```bash
python -c "import json; from pathlib import Path; from generate import generate; cfg=json.loads(Path('experiment.json').read_text()); cfg.update(nonzeros=4,snr_db=[20],clean_signals=dict(train=64,val=16,test=16)); generate(cfg,Path('data/s4'))"
python train.py train --model SALISTA --mode layerwise --snr 20 --data data/s4 --out runs/salista-s4
python train.py test  --model SALISTA --mode layerwise --snr 20 --data data/s4 --out runs/salista-s4
```

`data/s4` 是示例目录，可由现有 `generate.py` 配合 `nonzeros=4` 的实验配置生成。`--model all` 和默认网格仍为原有 45 个基线任务。已有绘图工具可显式选择 `--methods SALISTA`。模型工厂使用 `build_model("SALISTA", data, depth, config)`，其中 `config` 是含 `salista` 子对象的训练字典，高斯模式还需要 `snr_db`。`model(y,depth)` 返回 `[batch,n]`。`return_path=True` 返回逐层列表。启用 `return_diagnostics=True` 时额外返回逐层的步长、动量、阈值、下界、增量、坐标误差界和前后预算。训练器从 `run.json` 重建配置并加载 `best.pt`，诊断用不到真实支撑或标签。

1. **小证明例与回归。** 单位字典精确一步、非对称 Gram 行列方向、两种证书分支、零预算、投影、真支撑外抑制、批独立性、三头梯度、冻结与联合梯度、检查点回读均有测试。环境为 Python 3.12.14、PyTorch 2.8.0+cpu、NumPy 2.3.5。GPU 成本与算子行为需单独测量。
2. **首轮训练比较。** 固定同一 $A,W$，重新生成真实 $s=4$ 的配对划分。先比较共享 GRU 的两种日程，再分别更换单元、共享方式、动量或步长。训练种子建议 0、1、2 起，分开报告样本不确定性与种子间波动。匹配总更新数与样本暴露量，同时报告墙钟时间、峰值内存、参数量和证书耗时。各策略可选权重的验证次数也需记录。
3. **主实验。** 在真实 $s=50$ 的原划分上比较共享 GRU 的端到端和递增深度微调，同时保留严格模式对照。用最终检查点读取各深度。报告测试整体 NMSE、top-$s$ F1、实际支撑的误报/召回、全零率和缩放触发率，分别统计首次缩放前后的样本。将 $\rho$ 与步长、动量分别消融。如果研究目标要求全程有支撑保证，应优先研究在 $g\le0$ 时更紧且可验证的干扰上界。
