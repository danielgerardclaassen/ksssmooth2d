import autograd.numpy as np
import scipy.linalg as scilinalg
# import code.statfemtools.observetools as ot
import stfetools.observetools as ot
from scipy.special import gamma as gamma_func

def laplacian_spectral_eigenfunctions(n_indices_col_vec, x_points_col_vec, L):
    """
    Eigenfunctions of 1D Laplacian: (1/sqrt(Lt)) * sin(n*pi*(x+Lt)/(2*Lt))
    n_indices_col_vec: (m, 1)
    x_points_col_vec: (N, 1)
    Returns: (N, m) matrix
    """
    term_x = (x_points_col_vec + L) / (2 * L) # (N, 1)
    term_n_pi = n_indices_col_vec.T * np.pi # (1, m)
    arg_sin = term_x @ term_n_pi # (N,m) using broadcasting: (x_i+Lt)/(2Lt) * (n_j*pi)
    return (L**(-0.5)) * np.sin(arg_sin)

def laplacian_spectral_eigenvalues(j, L):
    return (np.pi*j / (2*L))**2

def matern_spectral_density_1d(omega, nu, ell, rho, logged=True):
    """
    Spectral density of the 1D Matérn covariance function.
    omega: frequency (scalar or array)
    nu: smoothness parameter
    length_scale: length scale parameter (l)
    sigma_f2: signal variance
    """
    if logged:
        ell = np.e**ell
        rho = np.e**rho

    if ell <= 0:
        raise ValueError("length_scale must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")

    # Normalization constant part
    const_num = 2 * np.sqrt(np.pi) * gamma_func(nu + 0.5) * (2 * nu)**nu
    const_den = gamma_func(nu) * ell**(2 * nu)
    constant = (rho**2) * const_num / const_den

    # Frequency-dependent part
    term_in_power = (2 * nu) / ell**2 + omega**2
    power_term = term_in_power**(-(nu + 0.5))
    return constant * power_term

def find_eigenvalue_cutoff_by_variance(eigenvalues_sorted_desc, variance_threshold=0.9975):
    """
    Determines the number of eigenvalues to keep to retain a certain
    percentage of the total variance.

    Args:
        eigenvalues_sorted_desc (np.ndarray): A 1D NumPy array of eigenvalues,
                                              sorted in descending order.
                                              These should be non-negative.
        variance_threshold (float): The desired proportion of total variance
                                    to retain (e.g., 0.9975 for 99.75%).

    Returns:
        int: The number of eigenvalues (modes) to keep.
        np.ndarray: The cumulative variance explained by keeping 1, 2, ..., N eigenvalues.
    """
    if not isinstance(eigenvalues_sorted_desc, np.ndarray):
        eigenvalues_sorted_desc = np.array(eigenvalues_sorted_desc)

    if np.any(eigenvalues_sorted_desc < -1e-9): # Allow for small numerical noise
        print("Warning: Some eigenvalues are negative. This method assumes non-negative eigenvalues (e.g., from a PSD matrix).")
        # Optionally, you could take absolute values or set negative ones to zero.
        # For this example, we proceed, but it's a sign of a potential issue with the input.

    # Ensure eigenvalues are non-negative for variance calculation
    # (especially important if coming from general SVD where singular values are always non-negative,
    # but eigenvalues from non-PSD matrices could be negative)
    positive_eigenvalues = np.maximum(eigenvalues_sorted_desc, 0)

    # 1. Calculate the sum of all (positive) eigenvalues
    total_variance = np.sum(positive_eigenvalues)

    if total_variance <= 0: # Handles cases with all zero or negative eigenvalues
        print("Warning: Total variance is zero or negative. Cannot determine cutoff meaningfully.")
        return 0, np.zeros_like(positive_eigenvalues) # Or raise an error

    # 2. Calculate the cumulative sum of the sorted (positive) eigenvalues
    cumulative_variance = np.cumsum(positive_eigenvalues)

    # 3. Normalize the cumulative sum by the total sum
    proportion_variance_explained = cumulative_variance / total_variance

    # 4. Find the first index k where this proportion exceeds the threshold
    num_eigenvalues_to_keep = np.argmax(proportion_variance_explained >= variance_threshold) + 1

    indices_meeting_threshold = np.where(proportion_variance_explained >= variance_threshold)[0]
    if len(indices_meeting_threshold) > 0:
        num_eigenvalues_to_keep = indices_meeting_threshold[0] + 1
    else:
        print(f"Warning: Could not meet variance threshold {variance_threshold}. "
              f"Max variance explained is {proportion_variance_explained[-1]:.4f} "
              f"with all {len(eigenvalues_sorted_desc)} eigenvalues.")
        num_eigenvalues_to_keep = len(eigenvalues_sorted_desc) # Keep all in this scenario

    return num_eigenvalues_to_keep, proportion_variance_explained

def build_hilbert_approx_kernel(xs, params, cutoff, lower=True, logged=True):
    L, nu, rho, ell = params
    mid_xs = (np.min(xs) + np.max(xs)) / 2
    xs_centered = xs - mid_xs
    basis_function_indices = np.arange(1, cutoff + 1).reshape(-1, 1)
    laplacian_eigenvalues = laplacian_spectral_eigenvalues(basis_function_indices, L=L)
    basis_functions = laplacian_spectral_eigenfunctions(basis_function_indices, xs_centered, L=L)
    frequencies_omega = np.sqrt(laplacian_eigenvalues)
    if logged:
        D_lambda_diag = matern_spectral_density_1d(frequencies_omega, nu=nu, ell=ell, rho=rho, logged=logged).flatten()
    else:
        D_lambda_diag = matern_spectral_density_1d(frequencies_omega, nu=nu, ell=ell, rho=rho).flatten()
    D_lambda_diag = np.maximum(D_lambda_diag, 1e-12)
    D_lambda_sqrt_diag = np.sqrt(D_lambda_diag)
    if lower:
        return basis_functions @ np.diag(D_lambda_sqrt_diag)
    else:
        return basis_functions @ np.diag(D_lambda_diag) @ basis_functions.T

def build_young_mirsky_approx_matrix(xs, params, cutoff, lower=True):
        xs_diff = build_dif_x(xs.reshape(-1,1))
        symmetric_matrix_M = K_theta_matrix(params, xs_diff, logged=False)
        # 1. Compute eigendecomposition
        # eigenvalues are sorted in ascending order by eigh
        # eigenvectors are columns of Q: Q[:, i] is eigenvector for eigenvalues[i]
        eigenvalues, Q = scilinalg.eigh(symmetric_matrix_M)

        # 2. Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1] # Indices for descending sort
        Lambda_sorted_desc = eigenvalues[sorted_indices]
        Q_sorted_desc = Q[:, sorted_indices]

        # 3. Truncate to k_modes
        Lambda_k_diag = Lambda_sorted_desc[:cutoff]
        Q_k = Q_sorted_desc[:, :cutoff]

        # Ensure eigenvalues are non-negative for PSD-like behavior (optional, good for kernel matrices)
        # Lambda_k_diag_nonneg = np.maximum(Lambda_k_diag, 0)
        # S_k_matrix = np.diag(Lambda_k_diag_nonneg) # This is essentially S_k from SVD

        S_k_matrix = np.diag(Lambda_k_diag)


        # 4. Reconstruct the low-rank matrix
        # M_k_approx = Q_k @ S_k_matrix @ Q_k.T
        # More efficiently:
        if lower:
            return Q_k @ np.diag(np.sqrt(Lambda_k_diag))
        else:
            return Q_k @ S_k_matrix @ Q_k.T

def logpdf_multivariate_normal(x, mean, cov):
    """
    Compute the log of the probability density function (PDF) of a multivariate normal distribution.

    Parameters:
    x : ndarray, shape (d,)
        The point at which to evaluate the logpdf.
    mean : ndarray, shape (d,)
        The mean vector of the distribution.
    cov : ndarray, shape (d, d)
        The covariance matrix of the distribution.

    Returns:
    logpdf : float
        The log-probability density of x.
    """
    d = mean.shape[0]  # Dimensionality
    cov_inv = np.linalg.inv(cov)  # Inverse of the covariance matrix
    cov_det = np.linalg.det(cov)  # Determinant of the covariance matrix

    diff = x - mean
    exponent = -0.5 * np.dot(diff.T, np.dot(cov_inv, diff))  # Quadratic form

    normalization = -0.5 * (d * np.log(2 * np.pi) + np.log(cov_det))

    return normalization + exponent

def build_dif_x(coordinates, coordinates_2=False):
    if coordinates_2 is False:
        num_spatial_points = len(coordinates)
        K_theta_1 = np.tile(coordinates, ((num_spatial_points, 1)))
        K_theta_2 = np.repeat(coordinates, (num_spatial_points), axis=0)
        return np.linalg.norm(K_theta_1 - K_theta_2, axis=1)
    else:
        num_spatial_points_1 = len(coordinates)
        num_spatial_points_2 = len(coordinates_2)
        K_theta_1 = np.tile(coordinates, ((num_spatial_points_2, 1)))
        K_theta_2 = np.repeat(coordinates_2, (num_spatial_points_1), axis=0)
        return np.linalg.norm(K_theta_1 - K_theta_2, axis=1)

def third_dif_matern(dif_x, rho, ell):
    C1 = np.sqrt(5)/ell
    C2 = 5/(3*(ell**2))
    C3 = -C1
    return (rho**2)*(1+C1*dif_x + C2*(dif_x**2))*np.exp(C3*dif_x)

def periodic_matern52(dif_x, rho, ell, period):
    """
    Computes the periodic Matérn 5/2 kernel.
    
    Args:
        dist_matrix: A matrix of pairwise distances, |x_i - x_j|.
        rho: The amplitude (variance) parameter.
        ell: The length scale parameter.
        period: The period of the domain (e.g., 32*pi).
    """
    # Wrap the distance on the circle
    d = np.minimum(dif_x, period - dif_x)
    
    # Standard Matérn 5/2 formula
    C1 = np.sqrt(5) / ell
    C2 = 5 / (3 * (ell**2))
    
    return (rho**2) * (1 + C1 * d + C2 * (d**2)) * np.exp(-C1 * d)

def d2_periodic_matern52_dx1_sq(dist_matrix, rho, ell, period):
    """
    Computes the second derivative of the periodic Matérn 5/2 kernel
    with respect to its first argument, x_i.
    
    k''(x_i, x_j) = ∂²k(x_i, x_j) / ∂x_i²
    """
    d = np.minimum(dist_matrix, period - dist_matrix)
    
    # The analytical second derivative of the Matérn 5/2 kernel
    C1 = 5 * rho**2 / (3 * ell**4)
    C2 = np.sqrt(5) / ell
    
    # Note: This derivative is taken assuming d = |x_i - x_j|, which is valid
    # almost everywhere. The non-differentiable points at the wrap-around
    # boundary have a negligible effect in practice.
    return C1 * (ell**2 - np.sqrt(5) * ell * d - 5 * d**2) * np.exp(-C2 * d)

def squared_exponential(dif_x, rho, ell):
    """
    Squared Exponential (RBF) kernel function.

    Parameters:
    - dif_x: distance (can be array-like)
    - rho: output scale (amplitude)
    - ell: length scale

    Returns:
    - Kernel value(s)
    """
    return (rho**2) * np.exp(-0.5 * (dif_x / ell)**2)

def K_theta_matrix(params, K_diff, kernel="matern", logged=True, shape=False, period=None):
    """
    Builds a covariance matrix from a kernel function.
    Now includes a 'period' argument for periodic kernels.
    """
    if shape is not False:
        n_dim_1, n_dim_2 = shape[0], shape[1]
    else:
        n_dim_1 = int(np.sqrt(len(K_diff)))
        n_dim_2 = n_dim_1
    match kernel:
        case "matern":
            if logged==True:
                rho, ell = np.e**params[0], np.e**params[1]
            else:
                rho, ell = params[0], params[1]
            k_flat = third_dif_matern(K_diff, rho, ell)
        case "periodic_matern":
            if period is None:
                raise ValueError("Period must be provided for periodic_matern kernel.")
            if logged==True:
                rho, ell = np.e**params[0], np.e**params[1]
            else:
                rho, ell = params[0], params[1]
            k_flat = periodic_matern52(K_diff, rho, ell, period)
        case "sqexp":
            if logged==True:
                rho, ell = np.e**params[0], np.e**params[1]
            else:
                rho, ell = params[0], params[1]
            k_flat = squared_exponential(K_diff, rho, ell)

    return np.reshape(k_flat, (n_dim_1, n_dim_2))

def update_mean_cov_gp(prediction_coordinates, prior_mean, prior_cov, 
                              observation_coordinates, observation_values, observation_covariance,
                              nugget_value=1e-9, max_nugget_retries=5):
    """
    Update GP mean and covariance using observations with numerical stability.
    Uses Cholesky decomposition and adds a nugget if the matrix is not positive definite.
    
    :param prediction_coordinates: The points X at which to compute the posterior (N_pred,).
    :param prior_mean: The prior mean at prediction_coordinates (N_pred,).
    :param prior_cov: The prior covariance K(X,X) at prediction_coordinates (N_pred, N_pred).
    :param observation_coordinates: The points X* where observations are made (N_obs,).
    :param observation_values: The observed values y* at observation_coordinates (N_obs,).
    :param observation_covariance: The observation noise covariance Sigma_n (N_obs, N_obs).
    :param nugget_value: Initial nugget to add to the matrix for inversion/solve.
    :param max_nugget_retries: Max attempts to increase nugget and retry Cholesky.
    :return: new_mean (N_pred,), new_cov (N_pred, N_pred)
    :raises: scilinalg.LinAlgError if Cholesky fails after retries.
    """
    prediction_coordinates = np.atleast_1d(prediction_coordinates)
    observation_coordinates = np.atleast_1d(observation_coordinates)
    
    n_pred = len(prediction_coordinates)
    n_obs = len(observation_coordinates)

    if n_obs == 0:
        # If no observations, posterior is the same as prior
        return prior_mean, prior_cov
        
    prior_mean = np.atleast_1d(prior_mean)
    prior_cov = np.atleast_2d(prior_cov)
    observation_values = np.atleast_1d(observation_values)
    observation_covariance = np.atleast_2d(observation_covariance)

    if prior_mean.shape != (n_pred,):
        raise ValueError("prior_mean must have shape (N_pred,).")
    if prior_cov.shape != (n_pred, n_pred):
        raise ValueError("prior_cov must have shape (N_pred, N_pred).")
    if observation_values.shape != (n_obs,):
         raise ValueError("observation_values must have shape (N_obs,).")
    if observation_covariance.shape != (n_obs, n_obs):
        raise ValueError("observation_covariance must have shape (N_obs, N_obs).")
    
    H_matrix = ot.build_absolute_observation_matrix_kdtree(
        prediction_coordinates.reshape(-1, 1), # X
        observation_coordinates.reshape(-1, 1)  # X*
    ) # Shape (N_obs, N_pred)
    K_pred_obs = prior_cov @ H_matrix.T # Shape (N_pred, N_obs)
    K_obs_obs_from_prior = H_matrix @ K_pred_obs # Shape (N_obs, N_obs)
    prior_mean_at_obs = H_matrix @ prior_mean # Shape (N_obs,)
    L = K_obs_obs_from_prior + observation_covariance # Shape (N_obs, N_obs)

    current_L = L.copy() # Work on a copy if adding nugget
    current_nugget = 0.0
    L_cho = None

    for attempt in range(max_nugget_retries + 1):
        try:
            if attempt > 0:
                # Add nugget if previous attempt failed
                current_nugget = nugget_value * (10**(attempt - 1)) # Increase nugget exponentially
                current_L = L + current_nugget * np.eye(n_obs)
            
            L_cho = scilinalg.cholesky(current_L, lower=True)

            break # Success!

        except scilinalg.LinAlgError as e:
            if attempt == max_nugget_retries:
                 error_message = (
                    f"Cholesky decomposition of L failed after {max_nugget_retries + 1} attempts, "
                    f"even with a nugget up to {current_nugget:.2e}. "
                    f"The matrix L (K(X*,X*) + Sigma_n) is likely not numerically positive definite. "
                    f"Original error: {e}"
                )
                 raise scilinalg.LinAlgError(error_message) from e
            # Continue to next attempt with larger nugget

    if L_cho is None: # Should not happen with the retry loop
         raise RuntimeError("Logic error in Cholesky stabilization loop.")

    diff_y = observation_values - prior_mean_at_obs # Shape (N_obs,)

    temp_solve_mean = scilinalg.solve_triangular(L_cho, diff_y, lower=True) # Shape (N_obs,)

    alpha = scilinalg.solve_triangular(L_cho.T, temp_solve_mean, lower=False) # Shape (N_obs,)

    new_mean = prior_mean + K_pred_obs @ alpha # Shape (N_pred,)

    temp_solve_cov = scilinalg.solve_triangular(L_cho, K_pred_obs.T, lower=True) # Shape (N_obs, N_pred)
    
    new_cov = prior_cov - temp_solve_cov.T @ temp_solve_cov # Shape (N_pred, N_pred)

    return new_mean, new_cov