import numpy as np

from scipy.interpolate import Rbf


def interpolate(points, colors, density, position: np.ndarray, p: float = 3) -> np.ndarray:
    distances = np.linalg.norm(points - position, axis=1)
    weights = 1 / (distances + (density * 0.25)) ** p

    numerator = np.sum(weights.reshape((-1, 1)) * colors, axis=0)
    denominator = np.sum(weights)

    return numerator / denominator


def rasterize_rbf(points, colors, size: tuple, function='linear', smooth: int = 15):
    width, height = size
    out_image = np.zeros((height, width, 3), dtype=np.float32)

    for c in [0, 1, 2]:
        try:
            rbf = Rbf(
                points[:, 0], points[:, 1],
                colors[:, c], function=function, smooth=smooth
            )

            grid_x, grid_y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
            grid_intensity = rbf(grid_x, grid_y)

        except ZeroDivisionError:
            grid_intensity = np.median(colors[:, c])

        out_image[:, :, c] = grid_intensity

    return np.clip(out_image, 0, 1)
