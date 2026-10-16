"""Train or evaluate a model-based L2O network.

The command-line flags match the original TensorFlow script, so ``commands.sh``
can be reused. Example::

    python train.py --model_name lista --task lasso --lasso_lam 0.005 \\
        --base_dir "$BASE_DIR" --data_dir "$BASE_DIR/data/25_50" \\
        --num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \\
        --exp_name Lista_lasso-0.005_m25_n50 --gpu 5
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine

MODEL_NAMES = (
    "lista",
    "lfista",
    "lamp",
    "step_lista",
    "lista_cp",
    "lista_cpss",
    "alista",
    "glista",
    "tista",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PyTorch model-based L2O trainer")
    parser.add_argument("--model_name", default="lista", choices=MODEL_NAMES)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--model_lam", type=float, default=0.4)
    parser.add_argument("--share_W", action="store_true")
    parser.add_argument("--task", default="sc", choices=("sc", "lasso", "cs"))
    parser.add_argument("--lasso_lam", type=float, default=0.005)
    parser.add_argument("--alista_W_file", default="W.npy")
    parser.add_argument("--glista_alti", type=float, default=5.0)
    parser.add_argument("--gain_func", default="relu", choices=("relu", "inv", "exp", "sigm", "none"))
    parser.add_argument("--tista_sigma2", type=float, default=None)
    parser.add_argument("--ss_q_per_layer", type=float, default=1.2)
    parser.add_argument("--ss_maxq", type=float, default=13.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_train_images", type=int, default=51200)
    parser.add_argument("--num_val_images", type=int, default=1024)
    parser.add_argument("--num_test_images", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--test_files", nargs="*", default=[])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--exp_name", default="lista_sc")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index. Default is 5 when that GPU is visible, otherwise 0.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    if args.num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    return engine.run(args)


if __name__ == "__main__":
    main()
