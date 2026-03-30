import numpy as np
import copy
import json
import argparse
import sys
import time
import os
import logging
import h5py

from skfem import *
from skfem.helpers import dot, grad
from tqdm import tqdm
import scipy.sparse as scisparse
import scipy.sparse.linalg as scisparselinalg
import scipy.linalg as scilinalg
from scipy.spatial import KDTree
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from scipy.interpolate import griddata

# Optional: High performance solver
try:
    from pypardiso import spsolve
    USING_PARDISO = True
except ImportError:
    from scipy.sparse.linalg import spsolve
    USING_PARDISO = False

# ==========================================
# 0. LOCAL TOOLS & UTILITIES
# ==========================================

def eval_math_expr(expr):
    if isinstance(expr, str):
        return eval(expr, {"__builtins__": None}, {"pi": np.pi, "np": np})
    return expr

def load_h5_dataset(filepath, groupname):
    """Loads observations from HDF5."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
        
    with h5py.File(filepath, 'r') as f:
        if groupname not in f:
            raise ValueError(f"Group {groupname} not found in {filepath}")
        g = f[groupname]
        data = {}
        for key in g.keys():
            data[key] = g[key][()]
    return data, {}

def matern_spectral_density_2d(omega_sq, nu, ell, rho):
    """
    Spectral density of 2D Matérn kernel.
    S(w) = C * (2nu/l^2 + w^2)^-(nu + d/2)
    """
    dim = 2
    
    coeff = (rho**2) * (4 * np.pi)**(dim / 2) * gamma_func(nu + dim / 2) / gamma_func(nu)
    coeff *= (2 * nu / ell**2)**nu
    
    base = (2 * nu / ell**2) + omega_sq
    return coeff * (base ** -(nu + dim / 2))

def get_2d_fourier_eigeninfo(rank, Lx, Ly):
    """
    Generates indices (kx, ky) and eigenvalues for a Rectangular Torus.
    """
    # 1. Generate a grid of integer wavenumbers
    # We need a pool large enough to find the lowest 'rank' eigenvalues.
    # Since aspect ratio is roughly 1.5:1 (40:27), a square box of integers is still fine (if big enough!)
    # provided it's large enough.
    K_max = int(np.sqrt(rank)) + 4 
    k_range = np.arange(-K_max, K_max + 1)
    kx, ky = np.meshgrid(k_range, k_range)
    
    kx = kx.flatten()
    ky = ky.flatten()
    
    # 2. Compute Eigenvalues (Rectangular Laplacian Spectrum)
    # The physical frequency depends on the specific dimension length
    freq_x = (2 * np.pi * kx) / Lx
    freq_y = (2 * np.pi * ky) / Ly
    
    eigenvalues = freq_x**2 + freq_y**2
    
    # 3. Sort by eigenvalue (low freq -> high freq)
    idx_sorted = np.argsort(eigenvalues)
    
    # 4. Keep top modes
    idx_keep = idx_sorted[:rank]
    
    return kx[idx_keep], ky[idx_keep], eigenvalues[idx_keep]

def eval_fourier_basis_1d(x, k, L):
    """Evaluates 1D Real Fourier Basis at points x for wavenumber k."""
    # Basis: 1/sqrt(L) if k=0
    #        sqrt(2/L) cos(...) if k > 0
    #        sqrt(2/L) sin(...) if k < 0  (Using sign of k to switch between cos/sin)
    
    # Precompute constants
    c0 = 1.0 / np.sqrt(L)
    cn = np.sqrt(2.0 / L)
    
    # Initialize output
    phi = np.zeros_like(x)
    
    # Case k = 0
    mask_0 = (k == 0)
    if np.any(mask_0):
        phi[:] = c0 # Broadcast to all x
        
    # Case k > 0 (Cosine)
    if k > 0:
        phi = cn * np.cos(2 * np.pi * k * x / L)
        
    # Case k < 0 (Sine)
    if k < 0:
        # Use abs(k) frequency, but Sine function
        phi = cn * np.sin(2 * np.pi * abs(k) * x / L)
        
    return phi

def build_periodic_hilbert_factor(basis_coords, params, rank, Lx, Ly):
    """
    Constructs the Low-Rank Factor L for a Rectangular Domain.
    """
    rho, ell = params[0], params[1]
    nu = 2.5 
    
    # 1. Get Rectangular Spectral Info
    kx_vec, ky_vec, lambda_vec = get_2d_fourier_eigeninfo(rank, Lx, Ly)
    
    # 2. Calculate Spectral Density
    # S(lambda) depends only on the eigenvalue magnitude
    spectral_densities = matern_spectral_density_2d(lambda_vec, nu, ell, rho)
    
    # 3. Evaluate Eigenfunctions
    x_coords = basis_coords[:, 0]
    y_coords = basis_coords[:, 1]
    
    n_nodes = len(x_coords)
    actual_rank = len(kx_vec)
    
    Phi = np.zeros((n_nodes, actual_rank))
    
    for i in range(actual_rank):
        # Evaluate X-basis using Lx
        val_x = eval_fourier_basis_1d(x_coords, kx_vec[i], Lx)
        
        # Evaluate Y-basis using Ly
        val_y = eval_fourier_basis_1d(y_coords, ky_vec[i], Ly)
        
        Phi[:, i] = val_x * val_y
        
    # 4. Construct L
    return Phi * np.sqrt(spectral_densities)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
parser = argparse.ArgumentParser(description="2D KS Low-Rank Filter/Smoother")
parser.add_argument("--config", type=str, required=True, help="Path to config")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

# check_stability_conditions(config)

PARENT_OPTS = config["PARENT_DATASET_OPTIONS"]
SIM_OPTS = config["SIMULATION_OPTIONS"]
SAVE_OPTS = config["SAVING_OPTIONS"]
LOG_OPTS = config.get("LOGGING_OPTIONS", {"ENABLED": False})
ADD_TAG = SAVE_OPTS.get("ADDITIONAL_TAG", "")
# Logging
logging.basicConfig(level=getattr(logging, LOG_OPTS["LEVEL"]),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=LOG_OPTS["FILEPATH"], filemode='w')
logging.getLogger('skfem').setLevel(logging.WARNING)
logger = logging.getLogger()

# Load Data
try:
    data, _ = load_h5_dataset(PARENT_OPTS["FILEPATH"], PARENT_OPTS["GROUPNAME"])
    obs_u_field = data[PARENT_OPTS["FIELD"]]
    obs_spatial = data[PARENT_OPTS["SPATIAL_COORDINATES"]]
    obs_temporal = data[PARENT_OPTS["TEMPORAL_COORDINATES"]]
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    sys.exit(1)

# Parameters
Lx = SIM_OPTS.get("DOMAIN_SIZE_X", 40.0)
Ly = SIM_OPTS.get("DOMAIN_SIZE_Y", 27.0)
DT = SIM_OPTS["DT"]
KAPPA = SIM_OPTS["MODEL_PARAMETERS"][0]
GP_PARAMS = SIM_OPTS["GP_PARAMETERS"]
LOW_RANK_K = int(GP_PARAMS[2])
SIGMA_OBS = GP_PARAMS[3]
ALPHA_AUG = SIM_OPTS["AUGMENTED_ALPHA"]
NEWTON_TOL = SIM_OPTS["NEWTON_TOL"]

# Time Setup
T_START = obs_temporal[0]
T_END = obs_temporal[-1]
num_time_points = int(round((T_END - T_START) / DT)) + 1
time_range = np.linspace(T_START, T_END, num_time_points)

# Map Observations to Simulation Steps (Nearest Neighbor)
obs_time_map = {}
logger.info("Mapping observations to simulation steps...")
count_mapped = 0
for obs_idx, t_obs in enumerate(obs_temporal):
    sim_idx = np.argmin(np.abs(time_range - t_obs))
    diff = np.abs(time_range[sim_idx] - t_obs)
    if diff <= DT * 0.5 + 1e-9:
        obs_time_map[sim_idx] = obs_idx
        count_mapped += 1
logger.info(f"Mapped {count_mapped} out of {len(obs_temporal)} observations.")

# ==========================================
# 2. FEM SETUP (Mixed Formulation)
# ==========================================
res_raw = SIM_OPTS["SPATIAL_RESOLUTION"]
if isinstance(res_raw, list):
    Nx, Ny = res_raw
else:
    Nx = Ny = res_raw

# Mesh Generation
mesh = MeshTri.init_tensor(
    np.linspace(0, Lx, Nx+1),
    np.linspace(0, Ly, Ny+1)
)
element = ElementTriP1()
basis = Basis(mesh, element)

dofs_left   = basis.get_dofs(lambda x: np.isclose(x[0], 0.0))
dofs_bottom = basis.get_dofs(lambda x: np.isclose(x[1], 0.0))

dofs_right = basis.get_dofs(lambda x: np.isclose(x[0], Lx))
dofs_top   = basis.get_dofs(lambda x: np.isclose(x[1], Ly))

n_nodes = basis.N
n_state = 2 * n_nodes # [u, w]

@BilinearForm
def M_mat(u, v, w): 
    return u * v

@BilinearForm
def A_mat(u, v, w): 
    return dot(u.grad, v.grad)

@BilinearForm
def A_aniso_mat(u, v, w):
    # Same Anisotropic Physics
    return -(1 - KAPPA) * u.grad[0] * v.grad[0] + KAPPA * u.grad[1] * v.grad[1]

@LinearForm
def F_nonlinear(v, w):
    return (w.u_1 * w.u_1.grad[0]) * v

@BilinearForm
def J_nonlinear(u, v, w):
    return (u * w.u_k.grad[0] + w.u_k * u.grad[0]) * v

def build_periodic_constraints(basis):
    idx_l = dofs_left.flatten()
    idx_r = dofs_right.flatten()
    idx_b = dofs_bottom.flatten()
    idx_t = dofs_top.flatten()
    
    coords = basis.doflocs
    y_l = coords[1, idx_l]; p_l = np.argsort(y_l); idx_l = idx_l[p_l]
    y_r = coords[1, idx_r]; p_r = np.argsort(y_r); idx_r = idx_r[p_r]
    x_b = coords[0, idx_b]; p_b = np.argsort(x_b); idx_b = idx_b[p_b]
    x_t = coords[0, idx_t]; p_t = np.argsort(x_t); idx_t = idx_t[p_t]
    
    row, col, val = [], [], []
    row_idx = 0
    def add_cons(idx1, idx2, offset):
        nonlocal row_idx
        for i, j in zip(idx1, idx2):
            row.extend([row_idx, row_idx]); col.extend([i + offset, j + offset]); val.extend([-1, 1]); row_idx += 1
            
    add_cons(idx_l, idx_r, 0)
    add_cons(idx_b, idx_t, 0)
    add_cons(idx_l, idx_r, n_nodes)
    add_cons(idx_b, idx_t, n_nodes)
    return scisparse.coo_matrix((val, (row, col)), shape=(row_idx, 2*n_nodes))

logger.info("Assembling constant matrices...")
M = M_mat.assemble(basis).tocsc()
A = A_mat.assemble(basis).tocsc()
A_aniso = A_aniso_mat.assemble(basis).tocsc()
C_csc = build_periodic_constraints(basis).tocsc()
CT_csc = C_csc.T.tocsc()
n_cons = C_csc.shape[0]
Reg_block = scisparse.identity(n_cons, format='csc') * -ALPHA_AUG
Zero_N = scisparse.csc_matrix((n_nodes, n_nodes))

# Linear Jacobian Parts (1/DT scaled)
# J11_const = (1/DT) * M 
# J12_const = -NU * A + M
J11_const = (1/DT) * M + A_aniso
J12_const = A
J21_const = A
J22_const = -M

J_const = scisparse.bmat([[J11_const, J12_const], [J21_const, J22_const]], format='csc')

# ==========================================
# 3. INITIALIZATION & HDF5 SETUP
# ==========================================
logger.info("Building Priors and Diffusion Matrices...")

# 1. Diffusion Matrix (Low Rank)
L_u = build_periodic_hilbert_factor(basis.doflocs.T, GP_PARAMS, LOW_RANK_K, Lx, Ly)

# Process noise applies to U only, W is auxiliary
L_Q = np.vstack([L_u, np.zeros_like(L_u)]) # (2N, Rank)
# M_Q = np.block([[M, np.zeros_like(M)], [np.zeros_like(M), np.zeros_like(M)]])
LOW_RANK_K = L_Q.shape[1] # Update in case Nystrom changed rank

# # 2. Initial State (Full [u, w])
curr_m = np.zeros(n_state)
curr_L = L_Q.copy()

# 3. Setup Disk Storage
if not os.path.exists(os.path.dirname(SAVE_OPTS["FILEPATH"])):
    os.makedirs(os.path.dirname(SAVE_OPTS["FILEPATH"]))

# f_h5 = h5py.File(SAVE_OPTS["FILEPATH"], 'w')
# group_name = args.config.split('/')[-1].replace('.json', '')
# if ADD_TAG != "":
#     group_name += ADD_TAG

# 1. Change mode to 'a' (Read/Write if exists, create otherwise)
f_h5 = h5py.File(SAVE_OPTS["FILEPATH"], 'a')

group_name = args.config.split('/')[-1].replace('.json', '')
if ADD_TAG != "":
    group_name += ADD_TAG

# 2. Check if this specific group exists to avoid a "Name already exists" error
#    or to fulfill your requirement of overwriting if it does exist.
if group_name in f_h5:
    del f_h5[group_name]

g = f_h5.create_group(group_name)

# Datasets - Storing FULL state to ensure Smoother math is correct
D_LAB = SAVE_OPTS["DATASET_LABELS"]
ds_m_pred = g.create_dataset(D_LAB["PRIOR"]["FIELD"], shape=(num_time_points, n_state), dtype='f4')
ds_L_pred = g.create_dataset(D_LAB["PRIOR"]["COVARIANCES"], shape=(num_time_points, n_state, LOW_RANK_K), dtype='f4')

ds_m_post = g.create_dataset(D_LAB["POSTERIOR"]["FIELD"], shape=(num_time_points, n_state), dtype='f4')
ds_L_post = g.create_dataset(D_LAB["POSTERIOR"]["COVARIANCES"], shape=(num_time_points, n_state, LOW_RANK_K), dtype='f4')

ds_m_smooth = g.create_dataset(D_LAB["SMOOTHED"]["FIELD"], shape=(num_time_points, n_state), dtype='f4')
ds_L_smooth = g.create_dataset(D_LAB["SMOOTHED"]["COVARIANCES"], shape=(num_time_points, n_state, LOW_RANK_K), dtype='f4')

g.create_dataset(D_LAB["SPATIAL_COORDINATES"], data=basis.doflocs.T)
g.create_dataset(D_LAB["TEMPORAL_COORDINATES"], data=time_range)

# --- FFBS Setup ---
num_ffbs = SIM_OPTS.get("NUM_FFBS_SAMPLES")
rng_ffbs = np.random.default_rng(SIM_OPTS.get("RANDOM_SEED"))

# Create dataset for samples: Shape (num_samples, T, 2N)
ds_ffbs = g.create_dataset("ffbs_samples", 
                           shape=(num_ffbs, num_time_points, n_state), 
                           dtype='f4', compression="gzip")

# Temporary buffer to hold the "current" state of all samples during the backward pass
# Shape: (num_samples, n_state)
samples_curr = np.zeros((num_ffbs, n_state))

# Observation Operator (KDTree)
tree = KDTree(basis.doflocs.T)
def get_H_matrix(obs_coords_t):
    _, indices = tree.query(obs_coords_t)
    n_obs = len(indices)
    row = np.arange(n_obs); col = indices; data = np.ones(n_obs)
    # H maps full state (2N) to observations. W is not observed.
    H_sub = scisparse.coo_matrix((data, (row, col)), shape=(n_obs, n_nodes))
    Zero_sub = scisparse.coo_matrix((n_obs, n_nodes))
    return scisparse.hstack([H_sub, Zero_sub], format='csr')

# ==========================================
# 4. FILTER LOOP (Write-to-Disk)
# ==========================================
logger.info(f"Starting Filter Loop: {len(time_range)} steps.")
pbar = tqdm(enumerate(time_range), total=len(time_range), desc="Filtering", leave=True)

for i, t in pbar:
    step_stats = {}
    
    # --- A. PREDICTION ---
    if i == 0:
        pred_m, pred_L = curr_m, curr_L
    else:
        # 1. Newton Solver
        guess = curr_m.copy()
        u_prev = curr_m[:n_nodes]
        for _ in range(10):
            u_k = guess[:n_nodes]
            w_k = guess[n_nodes:]
            
            # Physics Residuals
            r1 = (1/DT) * M @ (u_k - u_prev) + A_aniso @ u_k + A @ w_k
            r2 = - M @ w_k + A @ u_k

            r1_phys = F_nonlinear.assemble(basis, u_1=basis.interpolate(u_k))
            
            # Jacobian Assembly
            J_nl_val = J_nonlinear.assemble(basis, u_k=basis.interpolate(u_k))
            J_tot = J_const + scisparse.bmat([[J_nl_val, None], [None, Zero_N]], format='csc')
            
            # Augmented System
            g_cons = C_csc @ guess
            KKT = scisparse.bmat([[J_tot, CT_csc], [C_csc, Reg_block]], format='csc')
            RHS = np.concatenate([-(np.concatenate([r1 + r1_phys, r2])), -g_cons])
            
            delta = spsolve(KKT, RHS)[:n_state]
            guess += delta
            if np.linalg.norm(delta) < NEWTON_TOL: break
        pred_m = guess
        
        # 2. Covariance Propagation (Implicit, Normal Equations)        
        L_prev_u = curr_L[:n_nodes, :]
        
        # Construct RHS for propagation
        RHS_prop_top = (1/DT) * (M @ L_prev_u)
        RHS_prop = np.vstack([RHS_prop_top, np.zeros_like(L_prev_u), np.zeros((n_cons, LOW_RANK_K))])
        
        # Construct RHS for Process Noise
        L_Q_u = L_Q[:n_nodes, :] 

        # 2. Convert Field Noise -> Force Noise
        L_Q_force_u = M @ L_Q_u

        # 3. Reconstruct the Noise Vector
        #    Auxiliary variable w usually has no noise in KS formulation (-Delta u = w).
        #    Constraints have no noise.
        zeros_w = np.zeros((n_nodes, LOW_RANK_K))
        zeros_c = np.zeros((n_cons, LOW_RANK_K))

        # Stack: [Force_on_U, Force_on_W (0), Force_on_Constraints (0)]
        RHS_noise = np.vstack([L_Q_force_u, zeros_w, zeros_c])
        
        # Solves
        L_prop = spsolve(KKT, RHS_prop)[:n_state]
        L_noise = spsolve(KKT, RHS_noise)[:n_state]
        
        # Randomized SVD Truncation
        U, s, _ = randomized_svd(np.hstack([L_prop, L_noise]), n_components=LOW_RANK_K, random_state=42)
        pred_L = U @ np.diag(s)

    # WRITE PREDICTION
    ds_m_pred[i] = pred_m
    ds_L_pred[i] = pred_L

    # --- B. UPDATE ---
    if i in obs_time_map:
        idx = obs_time_map[i]
        y_obs = obs_u_field[idx]
        coords_t = obs_spatial[idx] if obs_spatial.ndim == 3 else obs_spatial
        
        H = get_H_matrix(coords_t)
        innov = y_obs - H @ pred_m
        
        HL = H @ pred_L
        S = HL @ HL.T + (SIGMA_OBS**2) * np.eye(len(y_obs))
        S_inv_innov = scilinalg.solve(S, innov, assume_a='pos')
        
        curr_m = pred_m + pred_L @ (HL.T @ S_inv_innov)
        
        # Update L using Cholesky of (I - KH)
        M_in = np.eye(LOW_RANK_K) - HL.T @ scilinalg.solve(S, HL, assume_a='pos')
        # Symmetrize
        M_in = (M_in + M_in.T)/2
        try:
            Z = scilinalg.cholesky(M_in, lower=False) # Returns Upper U such that U.T U = M
            curr_L = pred_L @ Z.T
            
            innov_norm = np.linalg.norm(innov)
            step_stats['innov'] = f"{innov_norm:.2e}"
            logger.info(f"UPDATE at Step {i}: innov={innov_norm:.2e}")
        except scilinalg.LinAlgError:
            curr_L = pred_L
            logger.warning(f"Step {i}: Cov Update failed.")
    else:
        curr_m, curr_L = pred_m, pred_L

    # WRITE POSTERIOR
    ds_m_post[i] = curr_m
    ds_L_post[i] = curr_L
    
    # Update Stats
    tr_P = np.sum(np.linalg.norm(curr_L, axis=0)**2)
    step_stats['tr_P'] = f"{tr_P:.2e}"
    pbar.set_postfix(step_stats)

# ==========================================
# 5. SMOOTHER LOOP (Low-Rank Optimized)
# ==========================================
logger.info("Starting Smoother...")

# Init last step
m_T = ds_m_smooth[-1] = ds_m_post[-1]
L_T = ds_L_smooth[-1] = ds_L_post[-1]

z_T = rng_ffbs.standard_normal((LOW_RANK_K, num_ffbs))
samples_curr = (m_T[:, None] + L_T @ z_T).T 
ds_ffbs[:, -1, :] = samples_curr

for i in tqdm(range(num_time_points-2, -1, -1), desc="Smoothing"):
    
    # 1. Load Data
    m_n = ds_m_post[i]
    L_n = ds_L_post[i]
    
    m_np1_pred = ds_m_pred[i+1]
    L_np1_pred = ds_L_pred[i+1]
    
    m_np1_smooth = ds_m_smooth[i+1]
    L_np1_smooth = ds_L_smooth[i+1]
    
    # 2. Recompute Jacobian J_{n+1}
    u_k = m_np1_pred[:n_nodes]
    J_nl_val = J_nonlinear.assemble(basis, u_k=basis.interpolate(u_k))
    J_tot = J_const + scisparse.bmat([[J_nl_val, None], [None, Zero_N]], format='csc')
    KKT_np1 = scisparse.bmat([[J_tot, CT_csc], [C_csc, Reg_block]], format='csc')

    # 3. Calculate Gain Matrix B_n (Low-Rank Way)
    # We need: B_n = L_n^T J_n^T J_{n+1}^{-T} (U S^-2 U^T)
    
    # SVD of P_{n+1|n}
    U, s, _ = scilinalg.svd(L_np1_pred, full_matrices=False)
    # Form the "Thin" RHS: U * S^-2. Shape (2N, K)
    # This is small (~40MB) compared to the full matrix (~3GB)
    Thin_RHS_Core = U @ np.diag(1.0 / (s**2 + 1e-12))
    
    # Augment for KKT solve (add zeros for constraints)
    # RHS shape: (2N + n_cons, K)
    RHS_adj = np.vstack([Thin_RHS_Core, np.zeros((n_cons, LOW_RANK_K))])
    
    # Solve Adjoint: Y_thin = J_{n+1}^{-T} @ (U S^-2)
    # Shape: (2N + n_cons, K)
    Y_aug = spsolve(KKT_np1.T, RHS_adj)
    Y_thin = Y_aug[:n_state, :] # (2N, K)
    
    # Apply J_n^T = -(1/DT) M
    # M acts on top block of Y_thin
    Y_thin_u = Y_thin[:n_nodes, :]
    Z_u = (1/DT) * (M @ Y_thin_u)
    Z_thin = np.vstack([Z_u, np.zeros_like(Y_thin[n_nodes:, :])]) # (2N, K)
    
    # Apply L_n^T
    # V = L_n^T @ Z_thin
    V = L_n.T @ Z_thin # (K, K) -- Very small!
    
    # Finally, reconstruct B_n
    # B_n = V @ U^T
    B_n = V @ U.T # (K, 2N)
    
    # 4. Update Mean
    diff_mean = m_np1_smooth - m_np1_pred
    # m_s = m_n + L_n @ (B_n @ diff)
    # Compute vector product first to stay O(N)
    correction_direction = B_n @ diff_mean
    m_smooth_val = m_n + L_n @ correction_direction
    
    ds_m_smooth[i] = m_smooth_val

    # 5. Update Covariance
    # M = I + (B_n L_s)(B_n L_s)^T - (B_n L_p)(B_n L_p)^T
    
    Q_s = B_n @ L_np1_smooth
    Q_p = B_n @ L_np1_pred
    
    M_inner = np.eye(LOW_RANK_K) + Q_s @ Q_s.T - Q_p @ Q_p.T

    diff_samples = (samples_curr - m_np1_pred).T # (2N, num_ffbs)
    m_cond = m_n[:, None] + L_n @ (B_n @ diff_samples) # (2N, num_ffbs)
    M_cond = np.eye(LOW_RANK_K) - Q_p @ Q_p.T
    Uz, Sz, _ = scilinalg.svd(M_cond)
    Z_cond = Uz @ np.diag(np.sqrt(np.maximum(Sz, 0.0)))
    eta = rng_ffbs.standard_normal((LOW_RANK_K, num_ffbs))
    samples_curr = (m_cond + L_n @ (Z_cond @ eta)).T # (num_ffbs, 2N)
    ds_ffbs[:, i, :] = samples_curr
    
    try:
        Uz, Sz, _ = scilinalg.svd(M_inner)
        Sz = np.maximum(Sz, 0.0)
        Z_n = Uz @ np.diag(np.sqrt(Sz)) @ Uz.T
        ds_L_smooth[i] = L_n @ Z_n
    except Exception as e:
        logger.warning(f"Smoother Cov SVD failed at step {i}: {e}. Keeping Filter Cov.")
        ds_L_smooth[i] = L_n

# Cleanup
f_h5.close()
logger.info("Complete.")