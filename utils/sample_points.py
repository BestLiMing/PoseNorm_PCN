# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import numpy as np


def sample_points(points: np.ndarray, n: int = 10000) -> np.ndarray:
    idx = np.random.choice(len(points), n, replace=len(points) < n)
    return points[idx]
