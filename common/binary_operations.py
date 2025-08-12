import typing

import numba
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import label, distance_transform_edt


@numba.jit(nopython=True)
def dilate(array: np.ndarray, iterations: int = 1, mask: np.ndarray = None) -> np.ndarray:
    result = np.zeros_like(array)
    height, width = array.shape

    for _ in range(iterations):
        for y, x in zip(*np.where(array)):
            result[y, x] = 1

            if y - 1 >= 0:
                result[y - 1, x] = 1
            if x - 1 >= 0:
                result[y, x - 1] = 1
            if y + 1 < height:
                result[y + 1, x] = 1
            if x + 1 < width:
                result[y, x + 1] = 1

    if mask is None:
        return result

    return np.logical_and(result, mask)


@numba.jit(nopython=True)
def count_true(binary_mask: np.ndarray) -> int:
    count = 0

    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                count += 1

    return count


@numba.jit(nopython=True)
def extract_segments(padded):
    pr, pc = padded.shape
    # We use a fixed-size array for segments; if needed, we overestimate and then trim.
    seg_arr = np.empty((pr * pc * 4, 4), dtype=np.int32)
    count = 0

    # Loop over pixels (skip border due to padding)
    for i in range(1, pr - 1):
        for j in range(1, pc - 1):
            if not padded[i, j]:
                continue

            # Top edge
            if not padded[i - 1, j]:
                # segment from (j - 1, i - 1) to (j, i - 1)
                seg_arr[count, 0] = j - 1
                seg_arr[count, 1] = i - 1
                seg_arr[count, 2] = j
                seg_arr[count, 3] = i - 1
                count += 1

            # Right edge
            if not padded[i, j + 1]:
                # segment from (j, i - 1) to (j, i)
                seg_arr[count, 0] = j
                seg_arr[count, 1] = i - 1
                seg_arr[count, 2] = j
                seg_arr[count, 3] = i
                count += 1

            # Bottom edge
            if not padded[i + 1, j]:
                # segment from (j - 1, i) to (j, i)
                seg_arr[count, 0] = j - 1
                seg_arr[count, 1] = i
                seg_arr[count, 2] = j
                seg_arr[count, 3] = i
                count += 1

            # Left edge
            if not padded[i, j - 1]:
                # segment from (j - 1, i - 1) to (j - 1, i)
                seg_arr[count, 0] = j - 1
                seg_arr[count, 1] = i - 1
                seg_arr[count, 2] = j - 1
                seg_arr[count, 3] = i
                count += 1

    return seg_arr[:count]


def build_edge_map(segments):
    # Use a dictionary to map each point to its connected segments.
    edge_map = {}
    for seg in segments:
        a = (seg[0], seg[1])
        b = (seg[2], seg[3])
        for pt in (a, b):
            if pt not in edge_map:
                edge_map[pt] = []
        edge_map[a].append((a, b))
        edge_map[b].append((a, b))
    return edge_map


def trace_polygon(edge_map):
    # Find the starting point: the one with minimal (y, x)
    all_points = list(edge_map.keys())
    start_point = min(all_points, key=lambda pt: (pt[1], pt[0]))
    polygon = [start_point]
    current_point = start_point
    segments_used = set()

    while True:
        found = False
        for seg in edge_map[current_point]:
            if seg in segments_used:
                continue
            a, b = seg
            next_point = b if a == current_point else a
            segments_used.add(seg)
            polygon.append(next_point)
            current_point = next_point
            found = True
            break
        if not found or current_point == start_point:
            break
    return polygon


def mask_to_polygon(mask, pad: int = 1):
    mask = mask.astype(np.bool_)
    padded = np.pad(mask, pad_width=pad, mode='constant', constant_values=False)
    segments = extract_segments(padded)
    if segments.shape[0] == 0:
        return []
    edge_map = build_edge_map(segments)
    polygon = trace_polygon(edge_map)
    return polygon


def split_at_choke_points(mask, max_erosion=5):
    mask = mask.astype(bool)  # Ensure binary format
    eroded = mask.copy()

    # Track removed pixels efficiently
    removed_pixels = np.zeros_like(mask, dtype=np.uint8)  # 8-bit storage
    for i in range(1, max_erosion + 1):
        new_eroded = binary_erosion(eroded)
        removed_pixels[eroded & ~new_eroded] = i  # Store erosion step
        if not new_eroded.any():  # Stop if fully eroded
            break
        eroded = new_eroded

    # Label remaining components
    labeled_components, num_components = label(eroded, structure=np.ones((3, 3), dtype=int))

    if num_components < 2:  # No choke points detected
        return mask.astype(np.int32), num_components  # Return as-is

    # Compute nearest surviving component for removed pixels
    dist, nearest_idx = distance_transform_edt(~eroded, return_indices=True)

    # Assign removed pixels to nearest component in a single pass
    assigned_mask = np.zeros_like(mask, dtype=np.int32)
    assigned_mask[eroded] = labeled_components[eroded]

    mask_nonzero = removed_pixels > 0  # Pixels that were eroded away
    assigned_mask[mask_nonzero] = labeled_components[nearest_idx[0][mask_nonzero], nearest_idx[1][mask_nonzero]]

    return assigned_mask, num_components


def iterative_erosion_tracking(binary_mask, min_area: int = 20):
    current_mask = binary_mask.copy()

    while np.any(current_mask):
        labeled_mask, num_components = label(current_mask)

        if num_components == 1:
            current_mask = binary_erosion(current_mask, iterations=1)

        else:
            result = []
            for i in range(1, num_components + 1):
                sub_mask = labeled_mask == i
                if np.count_nonzero(sub_mask) <= min_area:
                    continue

                result.extend(iterative_erosion_tracking(sub_mask, min_area=min_area))

            return result

    return [binary_mask]


def assign_nearest_mask(masks, mask):
    distances = np.full_like(mask, np.inf, dtype=np.float32)
    nearest_mask_idx = np.full(mask.shape, -1, dtype=int)

    for i, m in enumerate(masks):
        dist_map, indices = distance_transform_edt(~m, return_indices=True)
        update_mask = dist_map < distances  # Update if this mask is closer
        distances[update_mask] = dist_map[update_mask]
        nearest_mask_idx[update_mask] = i

    # Assign pixels from "mask" to their nearest mask in "masks"
    mask_indices = np.where(mask)
    for y, x in zip(*mask_indices):
        if nearest_mask_idx[y, x] != -1:  # Ensure a valid nearest mask exists
            masks[nearest_mask_idx[y, x], y, x] = True

    return masks


def assign_nearest_mask_floodfill(masks, mask):
    masks = np.array(masks, dtype=bool)  # Ensure boolean type
    mask = np.array(mask, dtype=bool)

    # Create a combined mask to track which pixels are assigned
    assigned = np.any(masks, axis=0)
    remaining = mask & ~assigned  # Pixels in "mask" not yet assigned

    # Define a structuring element for 4-connectivity
    structure = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=bool)

    last = 0
    while np.any(remaining):
        current = np.count_nonzero(remaining)
        if last == current:
            break

        last = current
        new_frontier = np.zeros_like(mask, dtype=bool)

        for i, m in enumerate(masks):
            border = binary_dilation(m, structure) & ~m  # Identify border pixels
            add = border & remaining  # Select valid pixels to be added
            masks[i] |= add  # Assign pixels to the mask
            new_frontier |= add  # Track newly assigned pixels

        remaining &= ~new_frontier  # Remove assigned pixels

    return masks


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, typing.Tuple[int, int, int, int]]:
    y_indices, x_indices = np.where(mask)
    if y_indices.size == 0 or x_indices.size == 0:
        raise ValueError("Mask contains no True values.")

    y_min, y_max = y_indices.min(), y_indices.max() + 1
    x_min, x_max = x_indices.min(), x_indices.max() + 1

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    return cropped_image, cropped_mask, (x_min, y_min, x_max, y_max)


def uncrop_array(cropped_array: np.ndarray, original_shape: tuple, bounding_box: tuple, fill_value=0) -> np.ndarray:
    x_min, y_min, x_max, y_max = bounding_box
    restored_array = np.full(original_shape, fill_value, dtype=cropped_array.dtype)

    restored_array[y_min:y_max, x_min:x_max] = cropped_array

    return restored_array


def apply_mask(label_array: np.ndarray, mask: np.ndarray):
    label_array[np.logical_not(mask)] = 0

    masked_values = label_array[mask]
    unique_vals, inverse = np.unique(masked_values, return_inverse=True)
    normalized = np.arange(1, len(unique_vals) + 1)

    label_array[mask] = normalized[inverse]



@numba.jit(nopython=True)
def lines_not_touching_mask(lines: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    n = lines.shape[0]
    result = np.ones(n, dtype=np.bool_)

    for i in range(n):
        x0 = int(lines[i, 0, 0])
        y0 = int(lines[i, 0, 1])
        x1 = int(lines[i, 1, 0])
        y1 = int(lines[i, 1, 1])

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            if 0 <= y0 < H and 0 <= x0 < W and mask[y0, x0]:
                result[i] = False
                break
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    return result
