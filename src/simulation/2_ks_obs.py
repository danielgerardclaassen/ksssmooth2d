import numpy as np
import json
import argparse
import h5py
import os
import copy
from scipy.spatial import cKDTree

# ==========================================
# 0. Helper Functions
# ==========================================
def load_h5_dataset(filepath, groupname):
    """Loads datasets and metadata attributes from HDF5."""
    with h5py.File(filepath, 'r') as f:
        if groupname not in f:
            raise ValueError(f"Group {groupname} not found in {filepath}")
        g = f[groupname]
        
        data = {}
        for key in g.keys():
            data[key] = g[key][()]
            
        # Load attributes (metadata)
        metadata = {}
        for key in g.attrs.keys():
            val = g.attrs[key]
            # Try to decode JSON strings back to dicts if possible
            try:
                metadata[key] = json.loads(val)
            except (TypeError, json.JSONDecodeError):
                metadata[key] = val
                
    return data, metadata

def save_to_h5(filepath, groupname, data_dict, metadata, overwrite=True):
    mode = 'a' # Append mode to keep file open if it exists
    if not os.path.exists(filepath):
        mode = 'w'
        
    with h5py.File(filepath, mode) as f:
        if groupname in f:
            if overwrite:
                del f[groupname]
            else:
                raise ValueError(f"Group {groupname} already exists.")
        
        g = f.create_group(groupname)
        
        for key, val in data_dict.items():
            g.create_dataset(key, data=val, compression="gzip")
            
        for m_key, m_val in metadata.items():
            if isinstance(m_val, dict):
                g.attrs[m_key] = json.dumps(m_val)
            else:
                g.attrs[m_key] = m_val
    print(f"Observations saved to {filepath} [{groupname}]")

def select_spatial_indices(node_coords, num_sensors, sampler_type, rng, domain_dims=None):
    """
    Selects indices of nodes to observe.
    """
    num_nodes = node_coords.shape[0]
    
    if sampler_type in ["static_random", "allrandom"]:
        return rng.choice(num_nodes, size=num_sensors, replace=False)
    
    elif sampler_type == "static_grid":
        # Calculate bounding box if not provided
        if domain_dims is None:
            min_x, min_y = np.min(node_coords, axis=0)
            max_x, max_y = np.max(node_coords, axis=0)
            Lx, Ly = max_x - min_x, max_y - min_y
        else:
            Lx, Ly = domain_dims
            # Assume mesh starts at coordinate min_x, min_y
            min_x, min_y = np.min(node_coords, axis=0)
            
        # Calculate aspect ratio
        aspect = Lx / Ly
        
        # Determine Grid Dimensions (nx, ny)
        nx = int(np.sqrt(num_sensors * aspect))
        ny = int(num_sensors / nx)
        
        # Adjust if we undershot
        while nx * ny < num_sensors:
            if Lx/nx > Ly/ny: nx += 1
            else: ny += 1
            
        print(f"Generating Periodic Sensor Grid: {nx} x {ny} (Target: {num_sensors})")
        
        # ============================================================
        # FIX FOR PERIODICITY:
        # Use endpoint=False. 
        # For a periodic domain of length L, the ideal points are:
        # 0, dx, 2dx, ..., (N-1)dx
        # The next point would be N*dx = L, which is implicitly 0.
        # ============================================================
        
        sx = np.linspace(min_x, min_x + Lx, nx, endpoint=False)
        sy = np.linspace(min_y, min_y + Ly, ny, endpoint=False)
        
        # OPTIONAL: Offset by half-stride.
        # This centers the sensors in the periodic "cells" rather than 
        # putting them exactly on the x=0 boundary line.
        # This is often numerically safer and avoids the question of 
        # "is the sensor at 0 or L?" entirely.
        dx = Lx / nx
        dy = Ly / ny
        sx += dx / 2.0
        sy += dy / 2.0
        
        XX, YY = np.meshgrid(sx, sy)
        target_points = np.column_stack([XX.ravel(), YY.ravel()])
        
        # Handle count mismatch
        if len(target_points) > num_sensors:
            print(f"Warning: Grid size ({len(target_points)}) > Requested ({num_sensors}). Trimming randomly.")
            indices_to_keep = rng.choice(len(target_points), num_sensors, replace=False)
            target_points = target_points[indices_to_keep]
        
        # Find nearest FEM nodes
        tree = cKDTree(node_coords)
        _, indices = tree.query(target_points)
        return np.unique(indices)
    
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    
# ==========================================
# 1. Setup
# ==========================================
parser = argparse.ArgumentParser(description="Make observations of 2D KS Equation")
parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

PARENT_OPTS = config["PARENT_DATASET_OPTIONS"]
OBS_OPTS = config["OBSERVATION_OPTIONS"]
SAVE_OPTS = config["SAVING_OPTIONS"]

# ==========================================
# 2. Load Parent Data (DGP)
# ==========================================
datasets, parent_meta = load_h5_dataset(PARENT_OPTS["FILEPATH"], PARENT_OPTS["GROUPNAME"])

dgp_field = datasets[PARENT_OPTS["FIELD"]]       # Shape (T, N)
dgp_coords = datasets[PARENT_OPTS["SPATIAL_COORDINATES"]] # Shape (N, 2)
dgp_times  = datasets[PARENT_OPTS["TEMPORAL_COORDINATES"]] # Shape (T,)

# Ensure dimensions match
if dgp_field.shape[1] != dgp_coords.shape[0]:
    # Handle case where coords might be transposed
    if dgp_field.shape[1] == dgp_coords.shape[1]:
        dgp_coords = dgp_coords.T
        
# ==========================================
# 3. Process Configuration
# ==========================================
# Inherit metadata
OBS_META = {
    "MODEL_NAME": parent_meta.get("MODEL_NAME", "Unknown"),
    "NOTES": parent_meta.get("NOTES", "") + " | Observed",
    "TYPE": "OBSERVATIONS",
    "PARAMETERS": copy.deepcopy(parent_meta.get("PARAMETERS", {}))
}

# Parameters
NUM_SPACE = OBS_OPTS["NUM_SPACE"]
NUM_TIME  = OBS_OPTS["NUM_TIME"]
NOISE_STD = OBS_OPTS["NOISE_STD"]
SAMPLER   = OBS_OPTS["SAMPLER"]
SEED      = OBS_OPTS["RANDOM_SEED"]
TIME_TRUNC = OBS_OPTS.get("TIME_TRUNCATION", 0.0)

TRI_TESTING = OBS_OPTS.get("TRI_TEST", False)

# Update Metadata
OBS_META["PARAMETERS"].update(OBS_OPTS)

rng = np.random.default_rng(SEED)

# ==========================================
# 4. Generate Observations
# ==========================================

# A. Time Slicing
# Filter out burn-in
valid_time_mask = dgp_times >= TIME_TRUNC
valid_times = dgp_times[valid_time_mask]
valid_field = dgp_field[valid_time_mask, :]

# Select equidistant time indices
# e.g., if we have 600 steps and want 60 obs, pick every 10th
if NUM_TIME > len(valid_times):
    raise ValueError(f"Requested {NUM_TIME} obs but only {len(valid_times)} steps available.")

time_indices = np.round(np.linspace(0, len(valid_times) - 1, NUM_TIME)).astype(int)
if TRI_TESTING:
    time_indices = np.round(np.linspace(1, len(valid_times) - 2, NUM_TIME)).astype(int)
    
obs_times = valid_times[time_indices]
obs_field_snapshots = valid_field[time_indices, :]

# for TRI testing, we need to have three datasets, the original, and then one shifted behind and the other ahead
if TRI_TESTING:
    DT = OBS_META["PARAMETERS"].get("DT", 0.25)
    time_indices_minus = time_indices - 1
    time_indices_plus = time_indices + 1
    obs_times_minus = obs_times - DT
    obs_times_plus = obs_times + DT
    obs_field_snapshots_minus = valid_field[time_indices_minus, :]
    obs_field_snapshots_plus = valid_field[time_indices_plus, :]

domain_dims = None
try:
    if "DOMAIN_SIZE_X" in OBS_META["PARAMETERS"]:
        Lx = float(OBS_META["PARAMETERS"]["DOMAIN_SIZE_X"])
        Ly = float(OBS_META["PARAMETERS"]["DOMAIN_SIZE_Y"])
        domain_dims = (Lx, Ly)
    elif "DOMAIN_SIZE_L" in OBS_META["PARAMETERS"]:
        # Fallback for square
        L = eval(str(OBS_META["PARAMETERS"]["DOMAIN_SIZE_L"]), 
                 {"__builtins__": None}, {"pi": np.pi})
        domain_dims = (L, L)
except:
    pass

# Container for results
obs_data = []
obs_coords_list = [] # For 'allrandom' case

if SAMPLER == "allrandom":
    if TRI_TESTING:
        raise ValueError("TRI Testing with 'allrandom' sampler is not supported due to inconsistent sensor locations.")
    # Different sensors every timestep
    for t in range(NUM_TIME):
        indices = select_spatial_indices(dgp_coords, NUM_SPACE, SAMPLER, rng, domain_dims)
        
        snapshot = obs_field_snapshots[t, indices]
        # Add noise
        noise = rng.normal(0, NOISE_STD, snapshot.shape)
        
        obs_data.append(snapshot + noise)
        obs_coords_list.append(dgp_coords[indices])
        
    obs_data = np.array(obs_data)
    # For allrandom, storing coords is tricky in HDF5 rectangular arrays.
    # We will store as (T, N_obs, 2)
    obs_spatial_coords = np.array(obs_coords_list)

else:
    # Static sensors (Random fixed or Grid fixed)
    indices = select_spatial_indices(dgp_coords, NUM_SPACE, SAMPLER, rng, domain_dims)
    
    # Slice all times at these indices
    # Shape (T_obs, N_sensors)
    truth_at_sensors = obs_field_snapshots[:, indices]
    noise = rng.normal(0, NOISE_STD, truth_at_sensors.shape)

    if TRI_TESTING:
        truth_at_sensors_minus = obs_field_snapshots_minus[:, indices]
        truth_at_sensors_plus = obs_field_snapshots_plus[:, indices]
        obs_data_minus = truth_at_sensors_minus + noise
        obs_data_plus = truth_at_sensors_plus + noise
    
    # Add Noise
    obs_data = truth_at_sensors + noise
    obs_spatial_coords = dgp_coords[indices]

# ==========================================
# 5. Saving
# ==========================================
if SAVE_OPTS["ENABLED"]:
    group_name = SAVE_OPTS.get("ADDITIONAL_TAG", "") 
    full_group_name = PARENT_OPTS["GROUPNAME"] + "_observed" + group_name
    
    # Handle the fact that spatial coords might be 2D (Static) or 3D (Allrandom)
    
    data_to_save = {
        SAVE_OPTS["DATASET_LABELS"]["FIELD"]: obs_data,
        SAVE_OPTS["DATASET_LABELS"]["TEMPORAL_COORDINATES"]: obs_times,
        SAVE_OPTS["DATASET_LABELS"]["SPATIAL_COORDINATES"]: obs_spatial_coords
    }
    
    if TRI_TESTING:
        data_to_save[ SAVE_OPTS["DATASET_LABELS"]["FIELD"] + "_minus" ] = obs_data_minus
        data_to_save[ SAVE_OPTS["DATASET_LABELS"]["FIELD"] + "_plus" ] = obs_data_plus
        data_to_save[ SAVE_OPTS["DATASET_LABELS"]["TEMPORAL_COORDINATES"] + "_minus" ] = obs_times_minus
        data_to_save[ SAVE_OPTS["DATASET_LABELS"]["TEMPORAL_COORDINATES"] + "_plus" ] = obs_times_plus

    save_to_h5(
        PARENT_OPTS["FILEPATH"], # Save into same file as DGP usually, or define new one
        full_group_name,
        data_to_save,
        OBS_META,
        overwrite=SAVE_OPTS["OVERWRITING"]
    )