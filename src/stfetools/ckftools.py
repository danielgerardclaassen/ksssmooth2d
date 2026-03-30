import numpy as np
import scipy.linalg as scilinalg
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

COND_THRESHOLD = 1e8
RR_KAPPA_MAX = 1e5

def fetch_sigma_points(mean_vector, covariance_matrix, nugget_value=1e-9, max_nugget_retries=5):
    """
    Generate sigma points for the Cubature Kalman Filter (CKF) based on the mean and covariance.
    Attempts to stabilize Cholesky decomposition by adding a small nugget if needed.
    
    :param mean_vector: The mean state vector (n,).
    :param covariance_matrix: The covariance matrix (n, n).
    :param nugget_value: Initial small value to add to the diagonal if Cholesky fails.
    :param max_nugget_retries: Maximum number of attempts to increase nugget and retry Cholesky.
    :return: The sigma points (n, 2*n) generated for the CKF.
    :raises: scilinalg.LinAlgError if Cholesky decomposition fails even after stabilization attempts.
    """
    mean_vector = np.atleast_1d(mean_vector)
    covariance_matrix = np.atleast_2d(covariance_matrix) # Ensures it's at least 2D for shape checks
    
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0

    if mean_vector.ndim != 1:
        raise ValueError("mean_vector must be a 1D array.")
    if covariance_matrix.shape != (mean_vector.shape[0], mean_vector.shape[0]):
        raise ValueError("covariance_matrix must have shape (n, n) matching mean_vector.")
    
    n_dims = mean_vector.shape[0]
    sqrt_n_dims = np.sqrt(n_dims)  # Scaling factor for sigma points
    
    if n_dims == 0: # Handle empty state
        return np.empty((0, 0))

    if n_dims == 1:
        # For 1D, covariance_matrix might be [[v]] or a scalar.
        # np.atleast_2d made it [[v]]. Get the scalar variance.
        scalar_variance = covariance_matrix[0, 0]
        if scalar_variance < 0:
            # Or try to fix by taking abs, or adding a nugget, or just erroring
            # For now, error if variance is negative.
            raise ValueError(f"1D covariance (variance) is negative: {scalar_variance}")
        
        # Handle case where variance is extremely close to zero due to constraints
        if np.isclose(scalar_variance, 0):
            sqrt_scalar_variance = 0.0
        else:
            sqrt_scalar_variance = np.sqrt(scalar_variance)
            
        # Ensure sqrt_covariance_matrix is array-like for consistent broadcasting later
        # The original .flatten() on np.sqrt(covariance_matrix) would produce [sqrt_scalar_variance]
        # So we achieve the same here for consistency if needed, though direct scalar math is fine for 1D.
        sqrt_cov_for_calc = np.array([sqrt_scalar_variance]) 

        sigma_points = np.array([
            mean_vector + sqrt_n_dims * sqrt_cov_for_calc,
            mean_vector - sqrt_n_dims * sqrt_cov_for_calc
        ]).T # Transpose to get shape (n_dims, 2*n_dims) -> (1, 2)
    else: # n_dims > 1
        if np.allclose(covariance_matrix, 0): # If matrix is effectively zero
            print("sqrt cov effectively zero, using base as sqrt")
            sqrt_covariance_matrix = covariance_matrix  # Cholesky of zero is zero
        else:
            current_covariance = covariance_matrix.copy() # Work on a copy
            current_nugget = nugget_value
            sqrt_covariance_matrix = None # Initialize

            for attempt in range(max_nugget_retries + 1): # +1 for the initial attempt without nugget
                try:
                    if attempt > 0: # If not the first attempt, add/increase nugget
                        # print(f"Warning: Cholesky attempt {attempt} failed. Adding nugget {current_nugget:.2e}.") # Optional: for debugging
                        # Add nugget to the original matrix to avoid accumulating nuggets
                        # if the original was already slightly perturbed.
                        # Or, add to current_covariance if you want incremental addition.
                        # For GP matrices, adding to the original is often preferred.
                        current_covariance = covariance_matrix + current_nugget * np.eye(n_dims)
                    
                    sqrt_covariance_matrix = scilinalg.cholesky(current_covariance, lower=True)
                    # If Cholesky succeeded and a nugget was added, you might want to log it.
                    if attempt > 0:
                        # print(f"Cholesky succeeded with nugget {current_nugget:.2e} on attempt {attempt+1}.") # Optional
                        pass
                    break # Successful Cholesky
                except scilinalg.LinAlgError as e:
                    if attempt == max_nugget_retries: # Last attempt failed
                        error_message = (
                            f"Cholesky decomposition failed after {max_nugget_retries + 1} attempts, "
                            f"even with a nugget up to {current_nugget:.2e}. "
                            f"The matrix is likely not numerically positive definite. Original error: {e}"
                        )
                        raise scilinalg.LinAlgError(error_message) from e
                    # Prepare for next attempt with a larger nugget
                    if attempt == 0: # First failure, nugget was 0 effectively
                        current_nugget = nugget_value # Start with the initial nugget_value
                    else:
                        current_nugget *= 10 # Increase nugget for the next try
            
            if sqrt_covariance_matrix is None: # Should not happen if loop logic is correct
                raise RuntimeError("Logic error in Cholesky stabilization loop.")

        unit_sigma_points = sqrt_n_dims * np.hstack((np.eye(n_dims), -np.eye(n_dims)))
        sigma_points = sqrt_covariance_matrix @ unit_sigma_points + mean_vector[:, np.newaxis]
    
    return sigma_points

def fetch_sigma_points_ckf_scaled(mean_vector, covariance_matrix,
                                  alpha_ckf=1.0, # New scaling parameter
                                  nugget_value=1e-9, max_nugget_retries=5):
    """
    Generate sigma points for a Cubature Kalman Filter (2n points)
    with a tunable spread parameter alpha_ckf.
    Implicitly assumes weights of 1/(2n) for mean and covariance calculations later.

    :param mean_vector: The mean state vector (n,).
    :param covariance_matrix: The covariance matrix (n, n).
    :param alpha_ckf: Scaling factor for sigma point spread.
                      alpha_ckf = 1.0 gives standard CKF (sqrt(n) scaling).
                      alpha_ckf < 1.0 pulls points closer.
                      alpha_ckf > 1.0 pushes points further.
    :param nugget_value: Initial small value to add to the diagonal if Cholesky fails.
    :param max_nugget_retries: Maximum number of attempts to increase nugget and retry Cholesky.
    :return: The sigma points (n, 2*n) generated.
    :raises: scilinalg.LinAlgError if Cholesky decomposition fails.
    """
    mean_vector = np.atleast_1d(mean_vector)
    covariance_matrix = np.atleast_2d(covariance_matrix)
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0

    if mean_vector.ndim != 1:
        raise ValueError("mean_vector must be a 1D array.")
    n_dims = mean_vector.shape[0]
    if covariance_matrix.shape != (n_dims, n_dims):
        raise ValueError("covariance_matrix must have shape (n, n) matching mean_vector.")

    if n_dims == 0:
        return np.empty((0, 0))

    # Determine scaling factor based on alpha_ckf and n_dims
    # The standard CKF uses sqrt(n_dims). We multiply this by alpha_ckf.
    base_scaling = np.sqrt(n_dims) if n_dims > 0 else 0.0
    effective_scaling_factor = alpha_ckf * base_scaling
    
    if n_dims == 0:
        return np.empty((0,0))

    # Cholesky decomposition (same as before)
    sqrt_P_matrix = None
    if np.allclose(covariance_matrix, 0):
        sqrt_P_matrix = np.zeros_like(covariance_matrix)
    else:
        current_cov_for_chol = covariance_matrix.copy()
        current_nugget = 0
        for attempt in range(max_nugget_retries + 1):
            try:
                if attempt > 0:
                    if attempt == 1: current_nugget = nugget_value
                    else: current_nugget *= 10
                    current_cov_for_chol = covariance_matrix + current_nugget * np.eye(n_dims)
                sqrt_P_matrix = scilinalg.cholesky(current_cov_for_chol, lower=True)
                break
            except scilinalg.LinAlgError as e:
                if attempt == max_nugget_retries:
                    raise scilinalg.LinAlgError(f"Cholesky failed: {e}") from e
        if sqrt_P_matrix is None: raise RuntimeError("Cholesky loop error.")

    # Generate sigma points
    if n_dims == 1:
        sqrt_scalar_variance = sqrt_P_matrix[0,0]
        # effective_scaling_factor for n_dims=1 is alpha_ckf * sqrt(1) = alpha_ckf
        sigma_points = np.array([
            mean_vector[0] + effective_scaling_factor * sqrt_scalar_variance,
            mean_vector[0] - effective_scaling_factor * sqrt_scalar_variance
        ]).reshape(1, 2)
    else: # n_dims > 1
        unit_sigma_vectors_scaled = np.eye(n_dims) * effective_scaling_factor
        sigma_points = np.zeros((n_dims, 2 * n_dims))
        for i in range(n_dims):
            sigma_points[:, i]        = mean_vector + sqrt_P_matrix @ unit_sigma_vectors_scaled[:, i]
            sigma_points[:, i + n_dims] = mean_vector - sqrt_P_matrix @ unit_sigma_vectors_scaled[:, i]
            
    return sigma_points

# def rr_regularise_matrix_test(matrix, condition_number_threshold = COND_THRESHOLD, rr_kappa_max = RR_KAPPA_MAX):
#     if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
#         print("!!! WARNING: Matrix input to rr_regularise_matrix_test contains NaN/Inf.")
#         # Return a default well-behaved matrix or raise an error

#     # Ensure matrix is symmetric before eigh
#     matrix_sym = (matrix + matrix.T) / 2.0

#     try:
#         eigenvals = np.linalg.eigh(matrix_sym)[0]
#     except np.linalg.LinAlgError as e:
#         print(f"!!! np.linalg.eigh failed in rr_regularise_matrix_test: {e}")
#         print("Matrix that failed eigh:")
#         print(matrix_sym)
#         # np.save("debug_eigh_fail_matrix.npy", matrix_sym) # Save for inspection
#         # Return a default well-behaved matrix or re-raise
    
#     l_min, l_max = eigenvals[0], eigenvals[-1]

#     cond_num = l_max / l_min if l_min > 0 else np.inf
#     if cond_num > condition_number_threshold:
#         target_min_eigenvalue = l_max / rr_kappa_max
#         rr_delta = max(0, target_min_eigenvalue - l_min)  # Ensure nonnegative shift
#         rr_delta = min(rr_delta, 1e-6 * np.trace(matrix) / matrix.shape[0])  # Limit magnitude
#         regularised_matrix = matrix +  rr_delta * np.eye(matrix.shape[0])
#         return regularised_matrix
#     else:
#         return matrix

      
def rr_regularise_matrix_test(matrix, condition_number_threshold = COND_THRESHOLD, rr_kappa_max = RR_KAPPA_MAX):
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        print("!!! WARNING: Matrix input to rr_regularise_matrix_test contains NaN/Inf.")
        # Return a default well-behaved matrix OR raise an error to stop earlier
        return np.eye(matrix.shape[0]) * 1e-6 # Example: return a small identity matrix
        # Or: raise ValueError("Input matrix to regularization contains NaN/Inf")

    matrix_sym = (matrix + matrix.T) / 2.0 # Ensure symmetry

    try:
        # eigenvals = np.linalg.eigh(matrix_sym)[0] # This only gets eigenvalues
        eigenvals, eigenvectors = np.linalg.eigh(matrix_sym) # Get both
    except np.linalg.LinAlgError as e:
        print(f"!!! np.linalg.eigh failed in rr_regularise_matrix_test: {e}")
        print("Matrix that failed eigh:")
        # print(matrix_sym) # Can be very large
        # Consider printing shape, min, max, if contains nan/inf again
        print(f"Shape: {matrix_sym.shape}, Has NaN: {np.any(np.isnan(matrix_sym))}, Has Inf: {np.any(np.isinf(matrix_sym))}")
        # np.save("debug_eigh_fail_matrix.npy", matrix_sym)
        return np.eye(matrix.shape[0]) * 1e-6 # Fallback

    l_min, l_max = eigenvals[0], eigenvals[-1] # Assumes eigenvals are sorted, eigh does this.

    # Handle cases where l_max might be zero or negative if matrix was zero or neg def.
    if l_max <= 1e-12: # Effectively zero or negative definite
        # Matrix is zero or negative definite. Simple ridge might be best.
        # Or force a small positive definite matrix.
        # This case indicates severe upstream problems.
        print(f"!!! WARNING: l_max <= 0 ({l_max:.2e}) in rr_regularise. Input matrix likely zero or neg-def.")
        return matrix_sym + 1e-6 * np.eye(matrix_sym.shape[0])


    cond_num = l_max / l_min if l_min > 1e-12 else np.inf # Avoid division by zero if l_min is tiny

    if cond_num > condition_number_threshold or l_min <= 0: # Added l_min <= 0 check
        target_min_eigenvalue = max(l_max / rr_kappa_max, 1e-9) # Ensure target is positive
        
        rr_delta = max(0, target_min_eigenvalue - l_min)
        
        # Original capping:
        # rr_delta = min(rr_delta, 1e-6 * np.trace(matrix_sym) / matrix_sym.shape[0])
        # This cap might be too restrictive if l_min is very negative.
        # A different cap, or just ensuring rr_delta isn't excessively huge:
        # max_allowed_trace_fraction_shift = 0.01 # e.g. shift by at most 1% of avg diagonal
        # cap_from_trace = max_allowed_trace_fraction_shift * np.abs(np.trace(matrix_sym) / matrix_sym.shape[0])
        # cap_from_trace = max(cap_from_trace, 1e-6) # ensure cap is at least some small positive
        # rr_delta = min(rr_delta, cap_from_trace)
        # A simpler approach if l_min is negative: ensure eigenvalues become target_min_eigenvalue
        
        # Simpler shift to ensure positivity if l_min is negative or too small:
        if l_min < target_min_eigenvalue:
            rr_delta = target_min_eigenvalue - l_min # This is the shift needed to make smallest eig target_min_eigenvalue
        else: # l_min is already good enough or matrix is well-conditioned
            rr_delta = 0

        if rr_delta > 0:
            # Reconstruct with shifted eigenvalues (more robust for ensuring PSD)
            new_eigenvals = np.maximum(eigenvals, target_min_eigenvalue)
            regularised_matrix = eigenvectors @ np.diag(new_eigenvals) @ eigenvectors.T
            return (regularised_matrix + regularised_matrix.T) / 2.0
            # Or simpler additive shift:
            # regularised_matrix = matrix_sym + rr_delta * np.eye(matrix_sym.shape[0])
            # print(f"Applied rr_delta: {rr_delta}")
            # return regularised_matrix
        else: # No regularization needed based on this logic path
            return matrix_sym
    else:
        return matrix_sym # Already well-conditioned

    

def propagate_sigma_points(model, sigma_points):
    """
    Propagate sigma points through a given model, which can be either
    a dynamical model (state transition) or an observational model.
    
    No numerical stability changes needed here, as it just applies the model.
    Potential issues are *within* the model function itself.

    :param model: A function that transforms each sigma point.
    :param sigma_points: The sigma points to be propagated (input_dim, num_points).
    :return: The propagated sigma points (output_dim, num_points), where output_dim
             is inferred dynamically from the model output.
    """
    sigma_points = np.atleast_2d(sigma_points)  # Ensure input is at least 2D
    input_dim, num_points = sigma_points.shape  

    if num_points == 0: # Handle empty input
        return np.empty((0, 0))

    # Evaluate the model on the first sigma point to infer output shape
    first_output = np.asarray(model(sigma_points[:, 0])).reshape(-1)
    output_dim = first_output.shape[0]

    # Ensure valid shape of first output
    if first_output.ndim != 1:
        raise ValueError(f"Model output must be 1D, but got shape {first_output.shape}")

    # Initialize propagated sigma points array
    propagated_sigma_points = np.zeros((output_dim, num_points), dtype=np.float64)

    # Propagate each sigma point through the model
    # Use enumerate for clearer indexing
    for i, point in enumerate(sigma_points.T): # Iterate over columns (sigma points)
        output_i = np.asarray(model(point)).reshape(-1)

        # Validate output shape
        if output_i.shape != (output_dim,):
             raise ValueError(
                f"Model output shape mismatch at index {i}: Expected ({output_dim},), got {output_i.shape}"
            )

        propagated_sigma_points[:, i] = output_i

    # Final assertion to ensure valid output shape
    if propagated_sigma_points.shape != (output_dim, num_points):
         raise RuntimeError( # Use RuntimeError as this check should pass if logic is correct
            f"Expected propagated sigma points shape ({output_dim}, {num_points}), got {propagated_sigma_points.shape}"
        )

    return propagated_sigma_points

def compute_prediction_distribution(propagated_sigma_points, noise_matrix):
    """
    Compute the predicted state mean and covariance from the propagated sigma points.

    Corrected: Denominator should be the number of sigma points (2*n_x), not the output dimension.
    The number of points is num_points = propagated_sigma_points.shape[1].

    :param propagated_sigma_points: The propagated sigma points (output_dim, num_points).
    :param noise_matrix: The additive noise covariance matrix (output_dim, output_dim).
                           In CKF observation update, this is R.
    :return: The predicted state mean (output_dim,) and predicted covariance (output_dim, output_dim).
    """
    # Ensure input is at least 2D
    propagated_sigma_points = np.atleast_2d(propagated_sigma_points)

    output_dim, num_points = propagated_sigma_points.shape

    if num_points == 0: # Handle empty input
         return np.empty((output_dim,)), np.empty((output_dim, output_dim)) # Return empty mean and cov

    # Infer the original state dimension (n_x) from the number of points (2*n_x)
    # Note: This relies on CKF having 2*n_x points. If using different sigma point schemes,
    # this might need the original state dimension explicitly passed or inferred differently.
    # CKF specific check:
    if num_points % 2 != 0 or num_points < 2:
         raise ValueError(f"Invalid number of sigma points for CKF: {num_points}. Expected 2*n for some n >= 1.")
    # original_state_dim = num_points // 2 # Not strictly needed for calculation, but good for assertion if desired

    # Ensure noise matrix has valid shape matching the *output* dimension
    noise_matrix = np.atleast_2d(noise_matrix)
    if noise_matrix.shape != (output_dim, output_dim):
        raise ValueError(f"Noise matrix must be ({output_dim}, {output_dim}), got {noise_matrix.shape}")

    # Compute the mean of the propagated sigma points (weighted average)
    # CKF weights are 1 / (2*n_x), which is 1 / num_points for all points
    prediction_mean = np.mean(propagated_sigma_points, axis=1)

    # Ensure mean has shape (output_dim,)
    if prediction_mean.shape != (output_dim,):
         # This should not happen if np.mean(..., axis=1) behaves as expected
         raise RuntimeError(f"Expected prediction mean shape ({output_dim},), got {prediction_mean.shape}")

    # Compute centered sigma points
    centered_sigma_points = propagated_sigma_points - prediction_mean[:, np.newaxis] # Shape (output_dim, num_points)

    # Compute the prediction covariance
    # The division should be by the number of points, num_points (which is 2*n_x)
    prediction_covariance = (centered_sigma_points @ centered_sigma_points.T) / num_points + noise_matrix
    # centered_sigma_points @ centered_sigma_points.T is (output_dim, num_points) @ (num_points, output_dim) -> (output_dim, output_dim)

    # Ensure covariance matrix has shape (output_dim, output_dim)
    if prediction_covariance.shape != (output_dim, output_dim):
        # This should not happen if matrix multiplication behaves as expected
        raise RuntimeError(
            f"Expected prediction covariance shape ({output_dim}, {output_dim}), got {prediction_covariance.shape}"
        )

    return prediction_mean, prediction_covariance

def compute_cross_cov_state_observed(sigma_points_state, prediction_mean_state, 
                                     observation_propagated_sigma_points, observation_prediction_mean):
    """
    Compute the cross-covariance between the predicted state and the observation.
    
    Improved: Simplified centering using broadcasting, removed unnecessary 1D case logic.
    Ensures calculation aligns with the number of sigma points.

    :param sigma_points_state: The sigma points for the state (n_state_dims, num_points).
    :param prediction_mean_state: The predicted state mean (n_state_dims,).
    :param observation_propagated_sigma_points: The propagated observation sigma points (n_obs_dims, num_points).
    :param observation_prediction_mean: The predicted observation mean (n_obs_dims,).
    :return: The cross-covariance matrix between the state and the observation (n_state_dims, n_obs_dims).
    """
    sigma_points_state = np.atleast_2d(sigma_points_state)
    observation_propagated_sigma_points = np.atleast_2d(observation_propagated_sigma_points)

    n_state_dims, num_points_state = sigma_points_state.shape
    n_obs_dims, num_points_obs = observation_propagated_sigma_points.shape

    # Assert that the number of sigma points is consistent
    if num_points_state != num_points_obs:
        raise ValueError(f"Number of state sigma points ({num_points_state}) must match number of observation sigma points ({num_points_obs}).")

    num_points = num_points_state # This is 2*n_x for CKF
    
    if num_points == 0: # Handle empty input
         return np.empty((n_state_dims, n_obs_dims))

    # Infer the original state dimension (n_x) for the denominator weight
    if num_points % 2 != 0 or num_points < 2:
         raise ValueError(f"Invalid number of sigma points for CKF: {num_points}. Expected 2*n for some n >= 1.")
    # original_state_dim = num_points // 2 # Not strictly needed for denominator calculation

    prediction_mean_state = np.atleast_1d(prediction_mean_state)
    observation_prediction_mean = np.atleast_1d(observation_prediction_mean)

    if prediction_mean_state.shape != (n_state_dims,):
         raise ValueError(f"prediction_mean_state must have shape ({n_state_dims},), got {prediction_mean_state.shape}")
    if observation_prediction_mean.shape != (n_obs_dims,):
         raise ValueError(f"observation_prediction_mean must have shape ({n_obs_dims},), got {observation_prediction_mean.shape}")

    # Center the sigma points using broadcasting
    centered_state_points = sigma_points_state - prediction_mean_state[:, np.newaxis] # Shape (n_state_dims, num_points)
    centered_observation_points = observation_propagated_sigma_points - observation_prediction_mean[:, np.newaxis] # Shape (n_obs_dims, num_points)
    
    # Compute the cross-covariance matrix
    # P_xz = sum_{i=1}^{num_points} w_i * (X_i - x_mean) * (Z_i - z_mean)^T
    # For CKF, w_i = 1 / num_points for all i
    cross_covariance = (centered_state_points @ centered_observation_points.T) / num_points
    # (n_state_dims, num_points) @ (num_points, n_obs_dims) -> (n_state_dims, n_obs_dims)

    return cross_covariance

def compute_filter_gain(cross_covariance, observation_prediction_covariance, 
                        nugget_value=1e-9, max_nugget_retries=5):
    """
    Compute the Kalman filter gain using the cross-covariance and the observation prediction covariance.
    
    Improved: Uses scipy.linalg.solve for numerical stability and includes optional nugget handling
              for the observation prediction covariance matrix.

    :param cross_covariance: The cross-covariance matrix between the state and the observation (n_state_dims, n_obs_dims).
    :param observation_prediction_covariance: The predicted observation covariance matrix (n_obs_dims, n_obs_dims).
    :param nugget_value: Initial nugget to add to observation_prediction_covariance if solve fails.
    :param max_nugget_retries: Max attempts to increase nugget and retry solve.
    :return: The filter gain matrix (n_state_dims, n_obs_dims).
    :raises: scilinalg.LinAlgError if solving fails after retries.
    """
    cross_covariance = np.atleast_2d(cross_covariance)
    observation_prediction_covariance = np.atleast_2d(observation_prediction_covariance)

    n_state_dims, n_obs_dims_cross = cross_covariance.shape
    n_obs_dims_cov1, n_obs_dims_cov2 = observation_prediction_covariance.shape

    if n_obs_dims_cross != n_obs_dims_cov1 or n_obs_dims_cov1 != n_obs_dims_cov2:
        raise ValueError(f"Dimension mismatch: Cross-covariance is {cross_covariance.shape}, Observation prediction covariance is {observation_prediction_covariance.shape}")
    
    if n_obs_dims_cross == 0: # Handle case with no observations
        return np.empty((n_state_dims, n_obs_dims_cross))

    # We need to compute K = P_xz @ inv(P_zz)
    # This is equivalent to solving P_zz @ K.T = P_xz.T for K.T
    # K = solve(P_zz, P_xz.T).T

    current_P_zz = observation_prediction_covariance.copy() # Work on a copy if adding nugget
    current_nugget = 0.0
    filter_gain = None

    for attempt in range(max_nugget_retries + 1):
        try:
            if attempt > 0:
                 # Add nugget if previous attempt failed
                current_nugget = nugget_value * (10**(attempt - 1)) # Increase nugget exponentially
                current_P_zz = observation_prediction_covariance + current_nugget * np.eye(n_obs_dims_cov1)
                # print(f"Warning: Solving for gain failed, attempt {attempt+1}. Adding nugget {current_nugget:.2e}.") # Optional debug

            # Solve the linear system P_zz @ K.T = P_xz.T
            # scilinalg.solve expects right-hand side as (..., N) or (..., N, M)
            # P_zz is (N_obs, N_obs), P_xz.T is (N_obs, N_state)
            K_T = scilinalg.solve(current_P_zz, cross_covariance.T) # Shape (n_obs_dims, n_state_dims)
            filter_gain = K_T.T # Transpose back to get K (n_state_dims, n_obs_dims)

            if attempt > 0:
                # print(f"Solving for gain succeeded with nugget {current_nugget:.2e} on attempt {attempt+1}.") # Optional debug
                pass # Success!

            break # Successful solve

        except scilinalg.LinAlgError as e:
             if attempt == max_nugget_retries:
                 error_message = (
                    f"Solving for filter gain failed after {max_nugget_retries + 1} attempts, "
                    f"even with a nugget up to {current_nugget:.2e}. "
                    f"The observation prediction covariance matrix is likely not numerically positive definite. "
                    f"Original error: {e}"
                )
                 raise scilinalg.LinAlgError(error_message) from e
            # Continue to next attempt with larger nugget
        except ValueError as e: # Catch potential ValueErrors from solve related to singular matrix
             if attempt == max_nugget_retries:
                  error_message = (
                    f"Solving for filter gain failed after {max_nugget_retries + 1} attempts (ValueError), "
                    f"even with a nugget up to {current_nugget:.2e}. "
                    f"The observation prediction covariance matrix is likely singular. "
                    f"Original error: {e}"
                )
                  raise scilinalg.LinAlgError(error_message) from e # Re-raise as LinAlgError for consistency
             # Continue to next attempt with larger nugget

    if filter_gain is None: # Should not happen with the retry loop
         raise RuntimeError("Logic error in filter gain stabilization loop.")

    return filter_gain


def compute_updated_distribution(predicted_mean, filter_gain, measurement, 
                                 observation_prediction_mean, predicted_covariance, 
                                 observation_prediction_covariance):
    """
    Update the state estimate using the Kalman filter equations.
    
    Kept the standard update formula. Joseph form is not a direct drop-in replacement.
    Numerical stability relies on the gain calculation and initial predicted_covariance.

    :param predicted_mean: The predicted state mean (n_state_dims,).
    :param filter_gain: The Kalman filter gain (n_state_dims, n_obs_dims).
    :param measurement: The current measurement (n_obs_dims,).
    :param observation_prediction_mean: The predicted observation mean (n_obs_dims,).
    :param predicted_covariance: The predicted state covariance (n_state_dims, n_state_dims).
    :param observation_prediction_covariance: The predicted observation covariance matrix (n_obs_dims, n_obs_dims).
                                             Note: This should include the observation noise R.
    :return: The updated state mean (n_state_dims,) and covariance (n_state_dims, n_state_dims).
    """
    predicted_mean = np.atleast_1d(predicted_mean)
    filter_gain = np.atleast_2d(filter_gain)
    measurement = np.atleast_1d(measurement)
    observation_prediction_mean = np.atleast_1d(observation_prediction_mean)
    predicted_covariance = np.atleast_2d(predicted_covariance)
    observation_prediction_covariance = np.atleast_2d(observation_prediction_covariance)

    n_state_dims = predicted_mean.shape[0]
    n_obs_dims = measurement.shape[0]

    # Basic shape checks
    if filter_gain.shape != (n_state_dims, n_obs_dims):
         raise ValueError(f"Filter gain shape mismatch: Expected ({n_state_dims}, {n_obs_dims}), got {filter_gain.shape}")
    # ... add checks for other shapes if necessary ...

    # Innovation: (z_k - z_{k|k-1})
    innovation = measurement - observation_prediction_mean # Shape (n_obs_dims,)

    # Mean Update: x_{k|k} = x_{k|k-1} + K (z_k - z_{k|k-1})
    updated_mean = predicted_mean + filter_gain @ innovation # Shape (n_state_dims,)

    # Covariance Update: P_{k|k} = P_{k|k-1} - K P_{zz} K^T
    # P_zz is observation_prediction_covariance
    
    # Compute K @ P_zz
    term_kpz = filter_gain @ observation_prediction_covariance # Shape (n_state_dims, n_obs_dims)
    
    # Compute (K @ P_zz) @ K.T = K @ P_zz @ K.T
    # term_kpz @ filter_gain.T is (n_state_dims, n_obs_dims) @ (n_obs_dims, n_state_dims) -> (n_state_dims, n_state_dims)
    covariance_reduction = term_kpz @ filter_gain.T

    updated_covariance = predicted_covariance - covariance_reduction # Shape (n_state_dims, n_state_dims)

    return updated_mean, updated_covariance
