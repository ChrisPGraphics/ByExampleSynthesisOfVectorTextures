import typing

import bridson
import numpy as np


def generate_gradient_field(
        colors: np.ndarray, density, size: typing.Tuple[int, int]
):
    control_points = np.array(bridson.poisson_disc_samples(*size, density))
    control_colors = colors[np.random.randint(0, len(colors), len(control_points))]

    return control_points, control_colors
