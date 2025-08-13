import logging
import typing

import numpy as np
import scipy.optimize as optimize
import scipy.spatial as spatial
import tqdm

import hierarchy_node
import interpolate.polygon_interpolation as polygon_interpolation
import interpolate.control_point as control_point


def assignment_from_cost_matrix(a_textons, b_textons, cost_matrix) -> dict:
    vertex_assignment = {}
    unassigned_a = a_textons.copy()
    unassigned_b = b_textons.copy()

    for assigned_a, assigned_b in zip(*optimize.linear_sum_assignment(cost_matrix)):
        if a_textons[assigned_a] not in vertex_assignment:
            vertex_assignment[a_textons[assigned_a]] = []

        vertex_assignment[a_textons[assigned_a]].append(b_textons[assigned_b])
        unassigned_a.remove(a_textons[assigned_a])
        unassigned_b.remove(b_textons[assigned_b])

    for unassigned in unassigned_a:
        index = a_textons.index(unassigned)
        assigned_index = np.argmin(cost_matrix[index])

        if a_textons[index] not in vertex_assignment:
            vertex_assignment[a_textons[index]] = []

        vertex_assignment[a_textons[index]].append(b_textons[assigned_index])

    for unassigned in unassigned_b:
        index = b_textons.index(unassigned)
        assigned_index = np.argmin(cost_matrix[:, index])
        vertex_assignment[a_textons[assigned_index]].append(unassigned)

    return vertex_assignment


def interpolation_assignment(a_textons, b_textons, silent: bool = False) -> typing.Tuple[
    typing.List[polygon_interpolation.PolygonInterpolation],
    typing.Dict[hierarchy_node.VectorNode, typing.List[hierarchy_node.VectorNode]]
]:
    if silent:
        def log(*_):
            pass

    else:
        def log(*args):
            logging.info(*args)

    log("Building cost matrix")
    a_centroids = np.array([obj.get_centroid() for obj in a_textons])
    b_centroids = np.array([obj.get_centroid() for obj in b_textons])

    cost_matrix = spatial.distance_matrix(a_centroids, b_centroids)

    log("Computing polygon assignment")
    vertex_assignment = assignment_from_cost_matrix(a_textons, b_textons, cost_matrix)

    log("Computing interpolation plan...")
    interpolations = []
    for initial_polygon, final_polygons in vertex_assignment.items():
        for final_polygon in final_polygons:
            interpolations.append(polygon_interpolation.PolygonInterpolation(
                initial_polygon.exterior,
                final_polygon.exterior
            ))
            interpolations[-1].initial_color = initial_polygon.color
            interpolations[-1].final_color = final_polygon.color

    return interpolations, vertex_assignment


def primary_texton_mapping(
        source_textons: hierarchy_node.VectorNode, destination_textons: hierarchy_node.VectorNode
) -> typing.List[polygon_interpolation.PolygonInterpolation]:

    interpolations, vertex_assignment = interpolation_assignment(source_textons.children, destination_textons.children)

    logging.info("Creating map for child textons...")
    for initial_polygon, final_polygons in tqdm.tqdm(vertex_assignment.items()):
        initial_children = initial_polygon.children
        if len(initial_children) == 0:
            initial_children = [initial_polygon.copy()]

        for final_polygon in final_polygons:
            final_children = final_polygon.children
            if len(final_children) == 0:
                final_children = [final_polygon.copy()]

            interpolations.extend(interpolation_assignment(initial_children, final_children, silent=True)[0])

    return interpolations


def gradient_point_assignment(source_texture, destination_texture) -> dict:
    control_a = [control_point.ControlPoint(point, color) for point, color in zip(source_texture.points, source_texture.colors)]
    control_b = [control_point.ControlPoint(point, color) for point, color in zip(destination_texture.points, destination_texture.colors)]

    cost_matrix = spatial.distance_matrix(source_texture.points, destination_texture.points)

    return assignment_from_cost_matrix(control_a, control_b, cost_matrix)
