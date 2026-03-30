import numpy as np
from scipy.spatial import cKDTree

def select_equidistant_indices(interior_coords, interior_indices, num_observations, include_boundary=True):
    """
    Selects `n` equidistant indices and their corresponding values from the input array.

    This function generates `n` indices that are evenly spaced across the range of the input array, 
    and retrieves the corresponding values from the array. The indices are selected using numpy's 
    linspace function, which ensures the spacing between indices is as even as possible.

    Args:
        eg_array (numpy.ndarray): The input array from which values are selected.
        n (int): The number of equidistant indices to select.

    Returns:
        tuple: A tuple containing:
            - equidistant_indices (numpy.ndarray): An array of `n` equidistant indices.
            - equidistant_coordinates (numpy.ndarray): An array of values from `eg_array` at the selected indices.

    Example:
        >>> eg_array = np.array([10, 20, 30, 40, 50, 60, 70])
        >>> select_equidistant_indices(eg_array, 3)
        (array([0, 3, 6]), array([10, 40, 70]))
    """
    if include_boundary:
        # Generate `n` equidistant indices between 0 and len(eg_array) - 1
        equidistant_indices = np.linspace(0, len(interior_indices) - 1, num_observations, dtype=int)
        
        # Retrieve the values from `eg_array` at the selected indices
        equidistant_coordinates = interior_coords[equidistant_indices]
        
        # Return the selected indices and their corresponding values
        return equidistant_indices, equidistant_coordinates
    else:
        # Generate `n` equidistant indices between 0 and len(eg_array) - 1
        equidistant_indices = np.linspace(0, len(interior_indices) - 1, num_observations+2, dtype=int)
        equidistant_indices = equidistant_indices[1:-1]
        # Retrieve the values from `eg_array` at the selected indices
        equidistant_coordinates = interior_coords[equidistant_indices]
        
        # Return the selected indices and their corresponding values
        return equidistant_indices, equidistant_coordinates

def farthest_point_sampler_2D(interior_coords, interior_indices, num_observations):
    """
    Selects `num_observations` points from `interior_coords` such that each point is the farthest 
    from the previously selected points, using a greedy approach in 2D.

    This function implements a farthest point sampling algorithm in 2D. The first point is selected arbitrarily, 
    and subsequent points are chosen based on maximizing the minimum distance to the set of previously selected points.

    Args:
        interior_coords (numpy.ndarray): A 2D array of shape (N, 2) representing the coordinates of N points in 2D space.
        interior_indices (numpy.ndarray): A 1D array of shape (N,) representing the indices of the points in `interior_coords`.
        num_observations (int): The number of points to select based on the farthest point sampling strategy.

    Returns:
        numpy.ndarray: An array of selected indices from `interior_indices` corresponding to the selected points.

    Example:
        >>> interior_coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        >>> interior_indices = np.array([0, 1, 2, 3, 4])
        >>> farthest_point_sampler_2D(interior_coords, interior_indices, 3)
        array([0, 4, 2])  # Example output of selected indices
    """

    # Step 2: Select the first point arbitrarily (e.g., first index)
    selected_indices = [-1]  # Placeholder index to handle the first selection

    # Step 3: Compute pairwise Euclidean distances between all points
    pairwise_distances = np.linalg.norm(interior_coords[:, None] - interior_coords[None, :], axis=-1)

    # Step 4: Initialize distance tracking (minimum distance to the selected set)
    # Initial minimum distances are set to the distance from the first selected point
    min_distances = pairwise_distances[selected_indices[0], :]

    # Select the remaining points
    for _ in range(num_observations - 1):
        # Pick the farthest unselected point by finding the index with the maximum minimum distance
        next_point = np.argmax(min_distances)
        selected_indices.append(next_point)

        # Update the minimum distances to reflect the new set of selected points
        min_distances = np.minimum(min_distances, pairwise_distances[next_point, :])

    # Return the indices of the selected points from the original `interior_indices`
    return interior_indices[np.array(selected_indices)]

def random_sampler(interior_coords, interior_indices, num_observations, rng=None):
    """
    Randomly selects `num_observations` points from `interior_indices` without replacement.

    This function selects a set of points randomly from `interior_indices`. The selection is done 
    without replacement, meaning that each point can only be selected once.

    Args:
        interior_coords (numpy.ndarray): A 2D array of shape (N, 2) representing the coordinates of N points in 2D space.
        interior_indices (numpy.ndarray): A 1D array of shape (N,) representing the indices of the points in `interior_coords`.
        num_observations (int): The number of points to randomly select from the `interior_indices`.
        rng (numpy.random.Generator, optional): A random number generator instance. If None, a new one is created.

    Returns:
        numpy.ndarray: An array of selected indices from `interior_indices` corresponding to the randomly selected points.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> interior_indices = np.array([0, 1, 2, 3, 4])
        >>> random_sampler(interior_coords, interior_indices, 3, rng)
        array([3, 1, 4])  # Example output of randomly selected indices
    """

    if rng is None:
        rng = np.random.default_rng()  # Create a new generator if none is provided

    shuffled_indices = rng.permutation(interior_indices)

    # Use the generator for controlled randomness
    return rng.choice(shuffled_indices, size=num_observations, replace=False)

def sample_interior_points_1D_indices(coordinates, bc_vals, num_observations_space, sampler="random", rng=None):
    """
    Samples a specified number of interior points from a 1D space, given boundary conditions and a sampling method.

    This function samples points from within a defined interior region of a 1D space. It first removes points 
    that lie outside a rectangular boundary specified by `bc_vals`. The remaining points are then sampled using 
    one of the specified sampling methods (`random` or `FPS` for farthest point sampling).

    Args:
        coordinates (numpy.ndarray): A 1D array of shape (N) representing the coordinates of N points in 1D space.
        bc_vals (list or tuple): A list or tuple of boundary values `[x_min, x_max]` that define the region from which to sample interior points.
        num_observations_space (int): The number of points to sample from the available interior points.
        sampler (str): The sampling method to use. Options are:
                       - "random": Random sampling.
                       - "equal": equidistant point selection.
                       Default is "random".
        rng (numpy.random.Generator, optional): A random number generator instance. If None, a new one is created.

    Returns:
        numpy.ndarray: An array of selected indices corresponding to the sampled points from the interior region.

    Raises:
        ValueError: If `num_observations_space` exceeds the number of available interior points after applying the boundary conditions.
        ValueError: If an unsupported sampler method is specified.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> coordinates = np.array([0, 1, 2, 3, 4])
        >>> bc_vals = [0, 3]
        >>> sample_interior_points_1D_indices(coordinates, bc_vals, 2, sampler="random", rng=rng)
        array([1])  # Example output of selected interior point indices
    """

    # Step 1: Create a mask for points that lie within the boundary conditions
    x = coordinates[:, 0]
    interior_mask = (x > bc_vals[0]) & (x < bc_vals[1])

    # Step 2: Apply the mask to get the coordinates and indices of interior points
    interior_coords = coordinates[interior_mask]
    interior_indices = np.where(interior_mask)[0]

    # Step 3: Ensure that there are enough available interior points
    num_available = len(interior_coords)
    if num_observations_space > num_available:
        raise ValueError("Requested more points than available after boundary pruning.")

    # Step 4: Use the selected sampler method to sample points
    match sampler:
        case "random":
            selected = random_sampler(interior_coords, interior_indices, num_observations_space, rng)
            return np.sort(selected)
        case "equal":
            c1, c2 = select_equidistant_indices(interior_coords, interior_indices, num_observations_space, include_boundary=False)
            return c1
        case _:
            raise ValueError("Sampler not supported.")

def sample_interior_points_2D_indices(coordinates, bc_vals, num_observations_space, sampler="random", rng=None):
    """
    Samples a specified number of interior points from a 2D space, given boundary conditions and a sampling method.

    This function samples points from within a defined interior region of a 2D space. It first removes points 
    that lie outside a rectangular boundary specified by `bc_vals`. The remaining points are then sampled using 
    one of the specified sampling methods (`random` or `FPS` for farthest point sampling).

    Args:
        coordinates (numpy.ndarray): A 2D array of shape (N, 2) representing the coordinates of N points in 2D space.
        bc_vals (list or tuple): A list or tuple of boundary values `[x_min, x_max, y_min, y_max]` that define 
                                  the rectangular region from which to sample interior points.
        num_observations_space (int): The number of points to sample from the available interior points.
        sampler (str): The sampling method to use. Options are:
                       - "random": Random sampling.
                       - "FPS": Farthest point sampling (FPS).
                       Default is "random".
        rng (numpy.random.Generator, optional): A random number generator instance. If None, a new one is created.

    Returns:
        numpy.ndarray: An array of selected indices corresponding to the sampled points from the interior region.

    Raises:
        ValueError: If `num_observations_space` exceeds the number of available interior points after applying the boundary conditions.
        ValueError: If an unsupported sampler method is specified.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> coordinates = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        >>> bc_vals = [0, 3, 0, 3]
        >>> sample_interior_points_2D_indices(coordinates, bc_vals, 2, sampler="random", rng=rng)
        array([1, 2])  # Example output of selected interior point indices
    """

    # Step 1: Create a mask for points that lie within the boundary conditions
    if coordinates.shape[1] != 2:
        raise ValueError('Coordinates have incorrect dimension, expected shape (N,2).')

    x, y = coordinates[:, 0], coordinates[:, 1]
    interior_mask = (x > bc_vals[0]) & (x < bc_vals[1]) & (y > bc_vals[2]) & (y < bc_vals[3])

    # Step 2: Apply the mask to get the coordinates and indices of interior points
    interior_coords = coordinates[interior_mask, :]
    interior_indices = np.where(interior_mask)[0]

    # Step 3: Ensure that there are enough available interior points
    num_available = len(interior_coords)
    if num_observations_space > num_available:
        raise ValueError("Requested more points than available after boundary pruning.")

    # Step 4: Use the selected sampler method to sample points
    match sampler:
        case "random":
            return random_sampler(interior_coords, interior_indices, num_observations_space, rng)
        case "FPS":
            return farthest_point_sampler_2D(interior_coords, interior_indices, num_observations_space)
        case _:
            raise ValueError("Sampler not supported.")

def build_observation_matrix_kdtree(basis_coordinates, observation_coordinates, k=3, eps=1e-12):
    """
    Constructs an observation matrix that maps from 'basis_coordinates' to 'observation_coordinates'
    using interpolation via k-nearest neighbors with inverse distance weighting (IDW).

    Parameters:
    - basis_coordinates: (n,) or (n, d) array of basis points (1D or d-dimensional).
    - observation_coordinates: (m,) or (m, d) array of observation locations.
    - k: Number of nearest neighbors to use for interpolation (default: 3).
    - eps: Small constant to avoid division by zero.

    Returns:
    - observation_matrix: (m, n) array that interpolates values from `basis_coordinates` to `observation_coordinates`.
    """
    # Ensure inputs are at least 2D
    basis_coordinates = np.atleast_2d(basis_coordinates)
    observation_coordinates = np.atleast_2d(observation_coordinates)

    # If 1D, reshape to column vectors (n,1) instead of (1,n)
    if basis_coordinates.shape[0] == 1:
        basis_coordinates = basis_coordinates.T
    if observation_coordinates.shape[0] == 1:
        observation_coordinates = observation_coordinates.T

    n, d1 = basis_coordinates.shape
    m, d2 = observation_coordinates.shape

    if d1 != d2:
        raise ValueError(f"Dimension mismatch: basis has {d1} dimensions, observations have {d2}.")

    observation_matrix = np.zeros((m, n), dtype=np.float64)

    # Build KDTree for fast nearest-neighbor search
    tree = cKDTree(basis_coordinates)

    # Query k nearest neighbors for each observation point
    k = min(k, n)  # Ensure k does not exceed available points
    distances, indices = tree.query(observation_coordinates, k=k)

    # Avoid division by zero (for exact matches)
    distances = np.maximum(distances, eps)

    # Compute inverse distance weights
    weights = 1.0 / distances
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize so weights sum to 1

    # Populate the observation matrix with weighted contributions
    for i in range(m):
        observation_matrix[i, indices[i]] = weights[i]

    return observation_matrix

def build_absolute_observation_matrix_kdtree(basis_coordinates, observation_coordinates):
    """
    Constructs an observation matrix that maps from 'basis_coordinates' to 'observation_coordinates'
    by selecting the single nearest neighbor.

    Parameters:
    - basis_coordinates: (n,) or (n, d) array of basis points (1D or d-dimensional).
    - observation_coordinates: (m,) or (m, d) array of observation locations.

    Returns:
    - observation_matrix: (m, n) binary array indicating nearest basis points for each observation.
    """
    # Ensure inputs are at least 2D
    basis_coordinates = np.atleast_2d(basis_coordinates)
    observation_coordinates = np.atleast_2d(observation_coordinates)

    # If 1D, reshape to column vectors (n,1) instead of (1,n)
    if basis_coordinates.shape[0] == 1:
        basis_coordinates = basis_coordinates.T
    if observation_coordinates.shape[0] == 1:
        observation_coordinates = observation_coordinates.T

    n, d1 = basis_coordinates.shape
    m, d2 = observation_coordinates.shape

    if d1 != d2:
        raise ValueError(f"Dimension mismatch: basis has {d1} dimensions, observations have {d2}.")

    observation_matrix = np.zeros((m, n), dtype=np.float64)

    # Build KDTree for fast nearest-neighbor search
    tree = cKDTree(basis_coordinates)

    # Query the single nearest neighbor for each observation point
    _, indices = tree.query(observation_coordinates, k=1)

    # Populate the observation matrix with exact matches
    for i in range(m):
        observation_matrix[i, indices[i]] = 1.0

    return observation_matrix