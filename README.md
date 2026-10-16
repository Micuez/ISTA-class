# Model-based L2O（PyTorch）

这里是 Open-L2O 中 model-based 可学习优化器的 PyTorch 实现，覆盖原来的 TensorFlow / Keras 版本：

LISTA、LFISTA、LAMP、Step-LISTA、LISTA-CP、LISTA-CPSS、ALISTA、GLISTA、TISTA。

训练流程与原脚本一致：按层展开网络，每一层先只更新本层新参数，再用两段更小的学习率微调更早的层。旧层梯度会乘上 `0.3` 的层龄次方。验证指标连续 5 个 epoch 没有下降就进入下一阶段。Adam 使用 Keras 的默认动量 `(0.9, 0.999)` 和 `eps=1e-7`。

## 环境

本机已经有 conda 环境 `openl2o`（Python 3.12，PyTorch 2.9.1，CUDA 12.8），对应 RTX 3090：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openl2o
```

这个环境复用了本机已经装好的 CUDA 12.8 PyTorch，避免再下一份数 GB 的 wheel。换机器时可以重建：

```bash
conda create -n openl2o python=3.12 -y
conda activate openl2o
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy
```

默认使用第 5 张可见 GPU（本机优先物理编号 5–9）。只有一张卡可见时用第 0 张。也可以显式指定：

```bash
python train.py ... --gpu 5
```

## 数据

稀疏编码 / Lasso 目录需要：

| 文件 | 形状 | 含义 |
| --- | --- | --- |
| `A.npy` | `(M, N)` | 测量矩阵 |
| `train_data.npy` | `(样本数, M+N)` | 每一行是 `[y, x]` |
| `val_data.npy` | 同上 | 验证集 |
| `test_data.npy` | 同上 | 测试集 |
| `W.npy` | `(M, N)` | 仅 ALISTA，解析权重 |

压缩感知任务改为读取 `A_128_512.npy` 和 `D_256_512.npy`。图像域 PSNR 测试在原脚本里没有接通，这一版同样在层训练结束后停止。

## 训练与测试

`commands.sh` 里的参数可以直接用。例如 25×50 的 Lasso：

```bash
conda activate openl2o
export BASE_DIR=/path/to/experiment
python train.py --model_name lista --task lasso --lasso_lam 0.005 \
    --base_dir $BASE_DIR --data_dir $BASE_DIR/data/25_50 \
    --num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
    --exp_name Lista_lasso-0.005_m25_n50 --gpu 5
```

测试：

```bash
python train.py --model_name lista --task lasso --lasso_lam 0.005 \
    --base_dir $BASE_DIR --data_dir $BASE_DIR/data/25_50 \
    --num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
    --exp_name Lista_lasso-0.005_m25_n50 \
    --test --test_files test_data.npy --gpu 5
```

每个完成的层会写到 `models/<exp_name>/replicate_<id>/layer_<k>/model.pt`。中断后用同一条训练命令会从最近的连续层接着训。每层最后一阶段的损失和验证指标追加到 `logs/<exp_name>/replicate_<id>/metrics.jsonl`。

Lasso 看验证集上的 `0.5||Ax-y||^2 + λ||x||_1`（越小越好）。稀疏编码看逐样本 NMSE（dB，越小越好）。

## 和原实现对齐的细节

- 稀疏编码的训练损失是整个 batch 上的 `sum((x̂-x)^2)/2`，不是均值。Lasso 的训练损失沿用原来的 `0.5 * tf.nn.l2_loss + λ||x||_1`。日志里的验证指标才是通常报告的目标。
- 关闭了 TF32，矩阵乘法走 FP32。
- LFISTA 的 `We` 在原代码里是同一个变量被重复使用，这里也是所有层共享。
- 未开启 `--share_W` 时，LISTA 的步长是常数 1，更新被吸收进每层的 `W`。
