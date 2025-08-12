import typing

import bridson
import numpy as np
import synthesis.source_map as source_maps


def generate_gradient_field(
        colors: np.ndarray, density, size: typing.Tuple[int, int],
        source_map: source_maps.SourceMap = None, color_ids: np.ndarray = None
):
    control_points = np.array(bridson.poisson_disc_samples(*size, density))
    if source_map is None:
        control_colors = colors[np.random.randint(0, len(colors), len(control_points))]

    else:
        control_colors = []
        for point in control_points:
            source_id = np.random.choice(
                source_map.map_count, p=source_map.get_distribution((int(point[0]), int(point[1])))
            )
            mask = color_ids == source_id
            color_options = colors[mask]
            color_index = np.random.randint(0, len(color_options))
            control_colors.append(color_options[color_index])

        control_colors = np.array(control_colors)

    return control_points, control_colors
