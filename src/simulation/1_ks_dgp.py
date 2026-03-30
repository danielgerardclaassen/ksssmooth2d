import numpy as np
import scipy.sparse as scisparse
from skfem import *
from skfem.helpers import dot, grad
from tqdm import tqdm
import json
import argparse
import h5py
import time
import sys

# Optional: High performance solver
try:
    from pypardiso import spsolve
    USING_PARDISO = True
    print("Using pypardiso solver.")
except ImportError:
    from scipy.sparse.linalg import spsolve
    USING_PARDISO = False
    print("Using scipy.sparse.linalg solver.")

# ==========================================
# 0. Utilities
# ==========================================
def eval_math_expr(expr):
    """Evaluates strings like '32*pi' safely."""
    if isinstance(expr, str):
        return eval(expr, {"__builtins__": None}, {"pi": np.pi, "np": np})
    return expr

def save_to_h5(filepath, groupname, data_dict, metadata, overwrite=True):
    """Saves dictionaries of numpy arrays to HDF5 without wiping the whole file."""
    # Ensure directory exists
    os_dir = os.path.dirname(filepath)
    if os_dir and not os.path.exists(os_dir):
        os.makedirs(os_dir)

    # ALWAYS use 'a' (append). This creates the file if missing, 
    # but preserves content if it exists.
    with h5py.File(filepath, 'a') as f:
        # Manage Group
        if groupname in f:
            if overwrite:
                print(f"Group '{groupname}' exists. Deleting old group...")
                del f[groupname]
            else:
                # If we aren't overwriting, raise error to prevent data loss
                raise ValueError(f"Group '{groupname}' already exists in {filepath}. Set overwrite=True to replace it.")
        
        g = f.create_group(groupname)
        
        # Save Data
        for key, val in data_dict.items():
            g.create_dataset(key, data=val, compression="gzip")
            
        # Save Metadata
        for m_key, m_val in metadata.items():
            # Handle dictionary metadata by dumping to JSON string
            if isinstance(m_val, dict):
                g.attrs[m_key] = json.dumps(m_val)
            else:
                g.attrs[m_key] = m_val
                
    print(f"Saved to {filepath} [{groupname}]")

import os

# ==========================================
# 1. Setup & Configuration
# ==========================================
parser = argparse.ArgumentParser(description="Run 2D KS equation DGP")
parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
args = parser.parse_args()

# Load Config
with open(args.config, "r") as f:
    config = json.load(f)

SIM_OPTS = config["SIMULATION_OPTIONS"]
SAVE_OPTS = config["SAVING_OPTIONS"]

# Parse Parameters
Lx = SIM_OPTS.get("DOMAIN_SIZE_X", 40.0)
Ly = SIM_OPTS.get("DOMAIN_SIZE_Y", 27.0)
DT = SIM_OPTS["DT"]
T_MAX = SIM_OPTS["T_MAX"]
KAPPA = SIM_OPTS["KAPPA"]
NEWTON_TOL = SIM_OPTS["NEWTON_TOL"]
NEWTON_MAX = SIM_OPTS["NEWTON_MAX_ITER"]
ALPHA = SIM_OPTS["AUGMENTED_ALPHA"]
SEED = SIM_OPTS["RANDOM_SEED"]

rng = np.random.default_rng(SEED)

# ==========================================
# 2. FEM Setup (Mixed Formulation)
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

# Boundary DOFs
dofs_left   = basis.get_dofs(lambda x: np.isclose(x[0], 0.0))
dofs_bottom = basis.get_dofs(lambda x: np.isclose(x[1], 0.0))

dofs_right = basis.get_dofs(lambda x: np.isclose(x[0], Lx))
dofs_top   = basis.get_dofs(lambda x: np.isclose(x[1], Ly))

dof_coords = basis.doflocs # (2, N_nodes)
x_nodes = dof_coords[0, :]
y_nodes = dof_coords[1, :]

# ==========================================
# 3. Forms & Matrices
# ==========================================
# Constant Matrices
@BilinearForm
def M_mat(u, v, w):
    return u * v

@BilinearForm
def A_mat(u, v, w):
    return dot(u.grad, v.grad)

@BilinearForm
def A_aniso_mat(u, v, w):
    return -(1 - KAPPA) * u.grad[0] * v.grad[0] + KAPPA * u.grad[1] * v.grad[1]

# DGP Nonlinear Physics (Includes Advection/Damping)
@LinearForm
def F_phys_DGP(v, w):
    return (w.u_1 * w.u_1.grad[0]) * v

@BilinearForm
def J_phys_DGP(u, v, w):
    # 1. Burgers Jacobian
    u_k = w.u_k
    u_k_x = w.u_k.grad[0]
    delta = u
    delta_x = u.grad[0]
    
    j_nonlin = (delta * u_k_x + u_k * delta_x) * v
    
    return j_nonlin

# Constraint Builder (Periodic)
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
    
    n_dofs = basis.N
    row, col, val = [], [], []
    row_idx = 0
    
    def add_cons(idx1, idx2, offset):
        nonlocal row_idx
        for i, j in zip(idx1, idx2):
            row.extend([row_idx, row_idx])
            col.extend([i + offset, j + offset])
            val.extend([-1, 1])
            row_idx += 1
            
    add_cons(idx_l, idx_r, 0)      # u L-R
    add_cons(idx_b, idx_t, 0)      # u B-T
    add_cons(idx_l, idx_r, n_dofs) # w L-R
    add_cons(idx_b, idx_t, n_dofs) # w B-T
    
    return scisparse.coo_matrix((val, (row, col)), shape=(row_idx, 2*n_dofs))

# ==========================================
# 4. Assembly (Constant Parts)
# ==========================================
print("Assembling constant matrices...")
M = M_mat.assemble(basis).tocsc()
A = A_mat.assemble(basis).tocsc()
A_aniso = A_aniso_mat.assemble(basis).tocsc()
C_matrix = build_periodic_constraints(basis)

C_csc = C_matrix.tocsc()
CT_csc = C_matrix.T.tocsc()

# Augmented System Constants
n_cons = C_matrix.shape[0]
n_dofs = 2 * basis.N
Reg_block = scisparse.identity(n_cons, format='csc') * -ALPHA
Zero_N = scisparse.csc_matrix((basis.N, basis.N))

J11_const = (1/DT) * M + A_aniso
J12_const = A
J21_const = A
J22_const = -M

J_const = scisparse.bmat([
    [J11_const, J12_const],
    [J21_const, J22_const]
], format='csc')

# ==========================================
# 5. Initialization
# ==========================================

x_nodes = basis.doflocs[0, :]
y_nodes = basis.doflocs[1, :]

term1 = np.cos(2 * np.pi * x_nodes / Lx)
term2 = np.cos(2 * np.pi * x_nodes / Lx + 2 * np.pi * y_nodes / Ly)
term3 = np.sin(4 * np.pi * x_nodes / Lx + 2 * np.pi * y_nodes / Ly)
term4 = np.sin(2 * np.pi * y_nodes / Ly)
term5 = np.sin(4 * np.pi * y_nodes / Ly)

u_init = 0.1 * (term1 + term2 + term3 + term4 + term5)

u_init -= np.mean(u_init)

state = np.concatenate([u_init, np.zeros_like(u_init)])

solutions = []
time_points = np.arange(0, T_MAX, DT)

# ==========================================
# 6. Time Loop (Fast Form)
# ==========================================
print(f"Starting simulation ({len(time_points)} steps)...")
start_time = time.time()

for step_idx in tqdm(range(len(time_points))):
    guess = state.copy()
    u_prev = state[:basis.N]
    
    for newton_iter in range(NEWTON_MAX):
        u_k = guess[:basis.N]
        w_k = guess[basis.N:]
        
        # 1. Residuals
        # Linear part (divided by DT)
        r1_lin = (1/DT) * M @ (u_k - u_prev) + A_aniso @ u_k + A @ w_k
        r2_lin = - M @ w_k + A @ u_k
        
        # Nonlinear Physics 
        r1_phys = F_phys_DGP.assemble(basis, u_1=basis.interpolate(u_k))
        
        R = np.concatenate([r1_lin + r1_phys, r2_lin])
        
        # 2. Jacobian Update
        J_phys_block = J_phys_DGP.assemble(basis, u_k=basis.interpolate(u_k))
        
        # Create block for U-U interaction and add to J_const
        J_phys_expanded = scisparse.bmat([
            [J_phys_block, None],
            [None,         Zero_N]
        ], format='csc')
        
        J_total = J_const + J_phys_expanded
        
        # 3. Augmented Solve
        g = C_csc @ guess
        KKT = scisparse.bmat([
            [J_total, CT_csc],
            [C_csc,   Reg_block]
        ], format='csc')
        RHS = np.concatenate([-R, -g])
        
        delta_aug = spsolve(KKT, RHS)
        delta = delta_aug[:n_dofs]
        
        guess += delta
        if np.linalg.norm(delta) < NEWTON_TOL:
            break
            
    state = guess
    # Store only u-field
    solutions.append(state[:basis.N])

# ==========================================
# 7. Saving
# ==========================================
if SAVE_OPTS["ENABLED"]:
    # Prepare metadata with parameters
    meta = SAVE_OPTS["METADATA"]
    meta["PARAMETERS"] = SIM_OPTS # Embed parameters in metadata
    
    data_to_save = {
        SAVE_OPTS["DATASET_LABELS"]["FIELD"]: np.array(solutions),
        SAVE_OPTS["DATASET_LABELS"]["TEMPORAL_COORDINATES"]: time_points,
        SAVE_OPTS["DATASET_LABELS"]["SPATIAL_COORDINATES"]: basis.doflocs.T
    }
    
    save_to_h5(
        SAVE_OPTS["FILEPATH"],
        SAVE_OPTS["GROUPNAME"],
        data_to_save,
        meta,
        overwrite=SAVE_OPTS["OVERWRITE"]
    )
    