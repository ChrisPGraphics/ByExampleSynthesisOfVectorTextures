import logging
import math
import os
import typing

import PIL.Image
import cv2
import numba
import numpy as np
import shapely

import synthesis.weights as weights_object
import hierarchy_node
import matplotlib.pyplot as plt



@numba.jit(nopython=True)
def sample_distribution(choices, probabilities, size=1):
    cumulative_probabilities = np.cumsum(probabilities)
    selected_indices = np.searchsorted(cumulative_probabilities, np.random.random(size), side='right')
    selected_indices = np.clip(selected_indices, 0, len(probabilities) - 1)

    return choices[selected_indices]


@numba.jit(nopython=True)
def sample_distribution_once(probabilities):
    cumulative_probabilities = np.cumsum(probabilities)
    selected_index = np.searchsorted(cumulative_probabilities, np.random.random())

    return selected_index



def count_connected_pixels(mask: np.ndarray, coords):
    temp_mask = np.zeros_like(mask, dtype=np.uint8)
    temp_mask[mask] = 1

    temp_mask[coords[::-1]] = 1

    num, temp_mask, _, _ = cv2.floodFill(temp_mask, None, (coords[0][0], coords[1][0]), 255, flags=4)

    return np.count_nonzero(np.logical_and(mask, temp_mask == 255))


def get_probability_distribution(mask):
    distro = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
    return distro / distro.sum()


def sample_probability_distribution(distro: np.ndarray, count: int = None) -> typing.Tuple[int, int]:
    p_flat = distro.ravel()
    ind = np.arange(len(p_flat))

    # random_index = np.random.choice(ind, size=count, p=p_flat)
    random_index = sample_distribution(ind, p_flat, count)

    rows, cols = distro.shape
    row = random_index // cols
    col = random_index % cols

    return col, row


def primary_texton_distro(
        polygons: hierarchy_node.VectorNode, size: typing.Tuple[int, int],
        placement_tries: int = 20, improvement_steps: int = 5, max_fails: int = math.inf,
        selection_probability_decay: float = 2, weights: weights_object.Weights = weights_object.Weights(),
        log_steps_directory: str = None, placed_descriptors: np.ndarray = None,
        initial_polygons: typing.List[hierarchy_node.VectorNode] = None, placed_polygons: np.array = None,
        source_map=None, source_dict=None
) -> hierarchy_node.VectorNode:
    """
    placed_descriptors map:
        0 = unknown
        -1 = should be empty

    """

    def place_polygon(placed_texton: hierarchy_node.VectorNode, centroid: tuple):
        placed_texton = placed_texton.copy()
        placed_texton.set_centroid(centroid)

        descriptor_center = placed_texton.descriptor.center

        top_left = int(centroid[0] - descriptor_center[0]), int(centroid[1] - descriptor_center[1])

        descriptor_shape = placed_texton.descriptor.descriptor.shape
        placed_descriptor = placed_texton.descriptor.descriptor

        y_range = np.arange(descriptor_shape[0]) + top_left[1]
        x_range = np.arange(descriptor_shape[1]) + top_left[0]

        valid_y = (y_range >= 0) & (y_range < size[1])
        valid_x = (x_range >= 0) & (x_range < size[0])

        y_grid, x_grid = np.meshgrid(np.where(valid_y)[0], np.where(valid_x)[0], indexing='ij')

        offset_y = y_grid + top_left[1]
        offset_x = x_grid + top_left[0]

        valid_positions = (placed_descriptor[y_grid, x_grid] != 0) & \
                          (placed_descriptors[offset_y, offset_x] == 0) & \
                          (placed_polygons[offset_y, offset_x] == 0)

        placed_descriptors[offset_y[valid_positions], offset_x[valid_positions]] = placed_descriptor[
            y_grid[valid_positions], x_grid[valid_positions]
        ]

        placed_texton.binary_rasterize(placed_descriptors, color=-1)
        placed_texton.binary_rasterize(placed_polygons, color=placed_texton.category)

        result.add_child(placed_texton)

    def score_placement(placed_texton: hierarchy_node.VectorNode, centroid: tuple):
        try:
            coords = placed_texton.get_raster_coords(centroid, x_lim=(0, size[0]), y_lim=(0, size[1]))
        except OverflowError:
            return -math.inf

        descriptor_pixels = placed_descriptors[coords[::-1]]
        polygon_pixels = placed_polygons[coords[::-1]]

        target_area = np.sum(descriptor_pixels == placed_texton.category)
        mismatched_area = np.sum((descriptor_pixels != placed_texton.category) & (descriptor_pixels > 0))

        if weights.missed_area_weight == 0:
            missed_area = 0

        else:
            missed_mask = placed_descriptors == placed_texton.category
            if initial_placed_descriptors is not None:
                missed_mask = np.logical_and(missed_mask, np.logical_not(initial_placed_descriptors))

            missed_area = count_connected_pixels(missed_mask, coords)
            missed_area -= target_area

        empty_area = np.sum(descriptor_pixels <= 0)

        same_overlap_area = np.sum(polygon_pixels == placed_texton.category)
        different_overlap_area = np.sum((polygon_pixels != placed_texton.category) & (polygon_pixels != 0))

        return (
                weights.empty_area_weight * empty_area +
                weights.missed_area_weight * missed_area +
                weights.mismatched_area_weight * mismatched_area +
                weights.target_area_weight * target_area +
                weights.same_overlap_area_weight * same_overlap_area +
                weights.different_overlap_area_weight * different_overlap_area
        )

    size = (int(size[0]), int(size[1]))
    result = hierarchy_node.VectorNode.from_rectangle(size, None, polygons.color)

    if placed_polygons is None:
        placed_polygons = np.zeros(size[::-1], dtype=int)
    else:
        placed_polygons = placed_polygons.copy()

    if placed_descriptors is None:
        initial_placed_descriptors = None
        placed_descriptors = np.zeros(size[::-1], dtype=int)

    else:
        initial_placed_descriptors = placed_descriptors.copy()

    if log_steps_directory is not None:
        os.makedirs(os.path.join(log_steps_directory, "polygons"), exist_ok=True)
        os.makedirs(os.path.join(log_steps_directory, "descriptors"), exist_ok=True)

    textons = polygons.children[:]
    texton_choices = len(textons)
    if texton_choices == 0:
        logging.fatal("No primary textons were extracted in analysis. Skipping synthesis of primary texton layer...")
        return result

    if initial_polygons is None:
        if source_map is not None:
            initial_source = np.random.choice(
                source_map.map_count, p=source_map.get_distribution((size[0] // 2, size[1] // 2))
            )

            initial_index = np.random.randint(0, len(source_dict[initial_source]))
            place_polygon(source_dict[initial_source][initial_index], (size[0] / 2, size[1] / 2))

        else:
            initial_index = np.random.randint(0, texton_choices)
            place_polygon(textons[initial_index], (size[0] / 2, size[1] / 2))

    else:
        for polygon in initial_polygons:
            place_polygon(polygon, polygon.get_centroid())

    unique_categories = list(np.unique([i.category for i in textons if i is not None]))
    categorized_texton_indices = [
        [textons.index(texton) for texton in textons if texton is not None and texton.category == category] for category in unique_categories
    ]
    texton_selection_probabilities = [np.ones(len(d)) / len(d) for d in categorized_texton_indices]

    iteration = 0
    fails = 0

    while True:
        iteration += 1
        location_mask = placed_descriptors > 0

        if not np.any(location_mask):
            logging.info("No proposed pixels are left! Terminating early.")
            break

        probability_distribution = get_probability_distribution(location_mask)

        best_score = -math.inf
        best_centroid = None
        best_texton_index = None
        best_decay_index = None
        best_texton = None

        center_pixel = sample_probability_distribution(probability_distribution, count=1)
        center_pixel = (center_pixel[0][0], center_pixel[1][0])
        category_color = placed_descriptors[*center_pixel[::-1]]

        try:
            category_index = unique_categories.index(category_color)
        except ValueError:
            if category_color != 0:
                logging.warning(
                    "No polygons in the inclusion zone are of category {0}! "
                    "Clearing all category {0} pixels".format(
                        category_color
                    )
                )

            placed_descriptors[placed_descriptors == category_color] = 0
            category_index = None
            continue

        for _ in range(placement_tries):
            if source_map is not None:
                source_id = np.random.choice(
                    source_map.map_count, p=source_map.get_distribution(center_pixel)
                )

                reselect_multiplier = np.ones(len(texton_selection_probabilities[category_index]))
                while True:
                    if len([i for i in source_dict[source_id] if i.category == category_color]) == 0:
                        texton_index = None
                        break

                    adjusted_distro = texton_selection_probabilities[category_index] * reselect_multiplier
                    adjusted_distro /= adjusted_distro.sum()

                    random_index = sample_distribution_once(adjusted_distro)

                    texton_index = categorized_texton_indices[category_index][random_index]
                    candidate_texton = textons[texton_index]
                    if candidate_texton.source_id == source_id:
                        break

                    reselect_multiplier[random_index] = 0

                if texton_index is None:
                    logging.warning("Source {} has no textons of category {}!".format(source_id, category_color))
                    placed_descriptors[*center_pixel[::-1]] = 0
                    break

                candidate_texton = texton_index

            else:
                texton_index = categorized_texton_indices[category_index][sample_distribution_once(
                    texton_selection_probabilities[category_index]
                )]

                candidate_texton = textons[texton_index]

            decay_index = texton_index
            score = score_placement(candidate_texton, center_pixel)

            if score >= best_score:
                best_score = score
                best_centroid = center_pixel
                best_decay_index = decay_index
                best_texton = candidate_texton

        if best_centroid is None:
            continue

        original_best_centroid = best_centroid
        checked_centroids = [best_centroid]
        for _ in range(improvement_steps):
            checked = 0

            for new_center_pixel in [
                tuple(best_centroid + np.array([0, 1])),
                tuple(best_centroid + np.array([0, -1])),
                tuple(best_centroid + np.array([1, 0])),
                tuple(best_centroid + np.array([-1, 0])),
            ]:
                if new_center_pixel in checked_centroids:
                    continue

                checked += 1
                checked_centroids.append(new_center_pixel)

                try:
                    new_score = score_placement(best_texton, new_center_pixel)
                except OverflowError:
                    continue

                if new_score >= best_score:
                    best_score = new_score
                    best_centroid = new_center_pixel

            if checked == 0:
                break

        try:
            placed_descriptors[
                best_texton.get_raster_coords(best_centroid, x_lim=(0, size[0]), y_lim=(0, size[1]))[::-1]
            ] = -1
        except OverflowError:
            continue

        placed_descriptors[*original_best_centroid[::-1]] = -1

        if best_score < 0:
            fails += 1

            if fails > max_fails:
                break

            logging.info(
                "Iteration: {:>4}, Best Score: {:>9.3f}, Consecutive Fails: {:>2}".format(
                    iteration, best_score, fails
                )
            )

        else:
            fails = 0
            place_polygon(best_texton, best_centroid)

            logging.info(
                "Iteration: {:>4}, Best Score: {:>9.3f}, Placed Polygons: {:>4}".format(
                    iteration, best_score, len(result.children)
                ))

            # if isinstance(best_texton_index, int):
            #     best_texton = textons[best_texton_index]
            # else:
            #     best_texton = best_texton_index

            category_index = unique_categories.index(best_texton.category)

            original_decay_index = categorized_texton_indices[category_index].index(best_decay_index)
            texton_selection_probabilities[category_index][original_decay_index] /= selection_probability_decay

            # texton_selection_probabilities[category_index] /= selection_probability_decay
            total_probability = texton_selection_probabilities[category_index].sum()
            texton_selection_probabilities[category_index] /= total_probability

        if log_steps_directory is not None:
            adjusted = (placed_polygons / np.max(unique_categories) * 255).astype(np.uint8)
            PIL.Image.fromarray(adjusted).save(
                os.path.join(log_steps_directory, "polygons", "{:0>4}.png".format(iteration))
            )

            adjusted = (placed_descriptors / (np.max(unique_categories) + 1) * 255).astype(np.uint8)
            PIL.Image.fromarray(adjusted).save(
                os.path.join(log_steps_directory, "descriptors", "{:0>4}.png".format(iteration))
            )

    return result


def global_density_cleanup(
        result: hierarchy_node.VectorNode, global_coverage: float, per_category_coverage: dict,
        exempt_categories: typing.List[int] = None, exempt_polygons: typing.List[hierarchy_node.VectorNode] = None
):
    if exempt_polygons is None:
        exempt_polygons = []

    if exempt_categories is None:
        exempt_categories = []

    total_area = result.get_area()

    categories, current_coverage_area = get_category_area(result)
    target_coverage_area = [per_category_coverage[c] for c in categories]
    current_global_coverage = get_coverage(result)

    last_removed_polygon = None

    while True:
        if current_global_coverage <= global_coverage:
            break

        category_options = []

        for category, target_area, current_area in zip(categories, target_coverage_area, current_coverage_area):
            if current_area <= target_area:
                continue

            if category in exempt_categories:
                continue

            category_options.append(category)

        if len(category_options) == 0:
            break

        selected_category = category_options[np.random.randint(len(category_options))]
        removal_candidates = [
            texton for texton in result.children
            if texton.category == selected_category and texton not in exempt_polygons
        ]

        if len(removal_candidates) == 0:
            exempt_categories.append(selected_category)
            continue

        removal_index = np.random.randint(len(removal_candidates))
        last_removed_polygon = removal_candidates[removal_index]
        result.remove_child(last_removed_polygon)

        coverage_index = np.where(categories == selected_category)[0][0]

        removed_coverage = last_removed_polygon.get_area() / total_area
        current_coverage_area[coverage_index] -= removed_coverage
        current_global_coverage -= removed_coverage

    if last_removed_polygon is not None and np.random.random() > 0.5:
        result.add_child(last_removed_polygon)


def get_category_area(distribution: hierarchy_node.VectorNode):
    bounding_area = distribution.as_shapely()

    categories = np.unique([i.category for i in distribution.children if i is not None])
    coverage_area = [
        bounding_area.intersection(
            shapely.unary_union([
                texton.as_shapely().buffer(0) for texton in distribution.children
                if texton is not None and texton.category == category
            ])
        )
        for category in categories
    ]

    total_area = bounding_area.area

    return categories, [polygon.area / total_area for polygon in coverage_area]


def get_coverage(distribution: hierarchy_node.VectorNode):
    union = shapely.unary_union(
        [texton.as_shapely().buffer(0) for texton in distribution.children]
    )

    backdrop = distribution.as_shapely()
    union = backdrop.intersection(union)

    return union.area / backdrop.area
