# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import torch
from os.path import exists


def load_checkpoints(model, checkpoint_path, device):
    assert exists(checkpoint_path), f"Checkpoints do not exist: {checkpoint_path}"
    with open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)
