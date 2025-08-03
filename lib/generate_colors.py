# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import numpy as np
import colorsys
import matplotlib.pyplot as plt


def generate_colors(n, method='hsv'):
    if n <= 0:
        return np.empty((0, 3))

    if method == 'jet':
        v = np.linspace(0, 1, n)
        four_value = 4 * v
        red = np.clip(np.minimum(four_value - 1.5, -four_value + 4.5), 0, 1)
        green = np.clip(np.minimum(four_value - 0.5, -four_value + 3.5), 0, 1)
        blue = np.clip(np.minimum(four_value + 0.5, -four_value + 2.5), 0, 1)
        return np.column_stack((red, green, blue))

    elif method == 'hsv':
        hues = np.linspace(0, 1, n, endpoint=False)
        hsv_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues])
        return hsv_colors

    else:
        raise ValueError("Invalid method. Choose 'jet' or 'hsv'.")


def generate_distinct_colors(n, avoid_close_hues=True):
    if n <= 0:
        return np.empty((0, 3))

    if avoid_close_hues and n >= 3:
        base_hues = [0.0, 0.333, 0.666]
        hues = np.linspace(0, 1, n + 1, endpoint=False)[:n]
        hues = (hues + 0.1) % 1
    else:
        hues = np.linspace(0, 1, n, endpoint=False)

    colors = []
    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append([r, g, b])
    return np.array(colors)


if __name__ == "__main__":
    n = 14
    colors_jet = generate_colors(n, method='jet')
    print(colors_jet)
    colors_hsv = generate_colors(n, method='hsv')
    print(colors_hsv)

    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    for i, (c_jet, c_hsv) in enumerate(zip(colors_jet, colors_hsv)):
        ax[0].plot(i, 0, 's', markersize=20, color=c_jet)
        ax[1].plot(i, 0, 's', markersize=20, color=c_hsv)
    ax[0].set_title('Jet 色标')
    ax[1].set_title('HSV 色环')
    plt.tight_layout()
    plt.show()
