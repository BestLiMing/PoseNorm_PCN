# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

from torchinfo import summary
import torch
import sys
import time
from os.path import join, dirname, basename, exists

sys.path.append(dirname(dirname(__file__)))
ROOT = dirname(dirname(__file__))
NUM_CLASSES = 3
NUM_PARTS = 14
NUM_OCCS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EXP_DIR = rf"{ROOT}/paper"


def summary_model(model, inputs):
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        summary(model, input_data=tuple(inputs))
    else:
        summary(model, input_data=inputs)


def inference_time(model, inputs, num_runs=10):
    model.eval()
    total_time = 0.0
    with torch.no_grad():
        if isinstance(inputs, (list, tuple)):
            model(*inputs)
        else:
            model(inputs)

    for _ in range(num_runs):
        with torch.no_grad():
            start_time = time.perf_counter()

            if isinstance(inputs, (list, tuple)):
                out = model(*inputs)
            else:
                out = model(inputs)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            total_time += end_time - start_time

    return total_time / num_runs
