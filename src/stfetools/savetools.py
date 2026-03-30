import h5py
import numpy as np
from datetime import datetime
import json
import math

def eval_math_expr(expr):
    """
    Safely evaluate a mathematical expression string using math and numpy.
    """
    allowed_names = {}
    allowed_names.update(math.__dict__)
    allowed_names.update({'np': np, 'numpy': np})
    return eval(expr, {"__builtins__": {}}, allowed_names)

def create_h5_storage(dataset_index, filepath, groupname, metadata):
    with h5py.File(filepath, 'a') as f:  # Use append mode

        # Create group only if it doesn't exist
        if groupname in f:
            grp = f[groupname]
        else:
            grp = f.create_group(groupname)

        def make_dataset(name, shape_1d):
            if name not in grp:
                return grp.create_dataset(
                    name,
                    shape=(0, *shape_1d),
                    maxshape=(None, *shape_1d),
                    chunks=(1, *shape_1d),
                    dtype='float64'
                )
            else:
                return grp[name]  # Already exists

        for name, shape in dataset_index.items():
            make_dataset(name, shape)

        for key, value in metadata.items():
            value = convert_to_serializable(value)
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            grp.attrs[key] = value

def append_to_dataset(dataset_name, new_array, filepath, groupname):
    """
    Appends a single array `new_array` to dataset `dataset_name` within group `groupname` in file `filepath`.
    The array must match the second+ dimensions of the dataset.
    """
    with h5py.File(filepath, 'a') as f:
        if groupname not in f:
            raise ValueError(f"Group '{groupname}' not found in file '{filepath}'")
        grp = f[groupname]
        if dataset_name not in grp:
             raise ValueError(f"Dataset '{dataset_name}' not found in group '{groupname}'")

        dset = grp[dataset_name]
        current_t = dset.shape[0]
        dset.resize(current_t + 1, axis=0)
        dset[current_t] = new_array # new_array should be (1, ...) or just (...) matching shape[1:]

def read_timestep(dataset_name, index, filepath, groupname):
    """
    Reads the data at a specific time index from the given dataset within the specified group.
    """
    with h5py.File(filepath, 'r') as f:
        if groupname not in f:
            raise ValueError(f"Group '{groupname}' not found in file '{filepath}'")
        grp = f[groupname]
        if dataset_name not in grp:
             raise ValueError(f"Dataset '{dataset_name}' not found in group '{groupname}'")

        dset = grp[dataset_name]
        if index < 0 or index >= dset.shape[0]:
            raise IndexError(f"Index {index} out of range for dataset '{dataset_name}' of length {dset.shape[0]}")
        return dset[index]

def convert_h5_to_float32(filepath):
    with h5py.File(filepath, 'r+') as f:
        # Iterate over all groups and datasets in the file
        print("Converting to float32...")
        for key in f:
            dataset = f[key]
            if isinstance(dataset, h5py.Dataset):
                # Convert the dataset to float32
                data = dataset[:].astype(np.float32)

                # Now create a new dataset with compression, replacing the old one
                del f[key]  # Delete the old dataset
                f.create_dataset(key, data=data, compression="gzip")  # Recreate it with gzip compression
        print("Converted to float32 and recompressed!")

def delete_group(filepath: str, group_name: str):
    """Deletes an entire group, including all datasets and metadata, from an HDF5 file.

    Args:
        filepath (str): Path to the HDF5 file.
        group_name (str): Name of the group to delete.
    """
    with h5py.File(filepath, "a") as hdf5_file:  # Open in append mode
        if group_name in hdf5_file:
            del hdf5_file[group_name]
            print(f"Deleted group: '{group_name}' and all its contents.")
        else:
            print(f"Group '{group_name}' does not exist in the file.")

def read_dataset_length(dataset_name, filepath, groupname):
    """
    Returns the length (number of entries along the first axis) of a dataset in a specified group.

    Args:
        filepath (str): Path to the HDF5 file.
        groupname (str): Name of the group containing the dataset.
        dataset_name (str): Name of the dataset.

    Returns:
        int: The length of the dataset (size along axis 0).
    """
    with h5py.File(filepath, 'r') as f:
        if groupname not in f:
            raise ValueError(f"Group '{groupname}' not found in file '{filepath}'")
        grp = f[groupname]
        if dataset_name not in grp:
            raise ValueError(f"Dataset '{dataset_name}' not found in group '{groupname}'")
        return grp[dataset_name].shape[0]

def delete_datasets_in_group(filepath: str, group_name: str):
    """Deletes all datasets under the specified group in an HDF5 file if they exist.

    Args:
        filepath (str): Path to the HDF5 file.
        group_name (str): Name of the group containing datasets to delete.
    """
    with h5py.File(filepath, "a") as hdf5_file:  # Open in append mode
        if group_name in hdf5_file:
            group = hdf5_file[group_name]
            datasets_to_delete = [name for name in group if isinstance(group[name], h5py.Dataset)]
            
            for dataset_name in datasets_to_delete:
                del group[dataset_name]
                print(f"Deleted dataset: {group_name}/{dataset_name}")
        else:
            print(f"Group '{group_name}' does not exist in the file.")

def convert_to_serializable(value):
    """
    Convert a value to a JSON-serializable format.
    Specifically converts NumPy ndarrays to lists.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()  # Convert ndarray to list
    elif isinstance(value, (dict, list)):  
        # Recursively apply conversion to nested dictionaries/lists
        return recursive_convert(value)
    else:
        return value

def recursive_convert(value):
    """
    Recursively converts NumPy arrays inside dictionaries or lists to lists.
    """
    if isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_serializable(v) for v in value]
    else:
        return value

def save_field_to_file(file_path, group_name, data_dict, metadata=None, overwrite=None):
    """
    Saves data to an HDF5 file under a specified group. Allows appending of new datasets, overwriting of existing ones, 
    and inclusion of metadata with the saved datasets.

    Args:
        file_path (str): The path to the HDF5 file where data will be saved.
        group_name (str): The group name within the HDF5 file to store the data.
        data_dict (dict): A dictionary where keys are dataset names and values are the corresponding data to store.
        metadata (dict, optional): A dictionary containing metadata to be stored as attributes in the group. Default is None.
        overwrite (bool, optional): If True, existing datasets will be overwritten. If False, existing datasets will be skipped. Default is None.

    Returns:
        None
    """
    # Default mode is append (this prevents overwriting the file entirely)
    mode = "a"  # Append mode
    with h5py.File(file_path, mode) as f:
        # Get or create the specified group within the HDF5 file
        group = f.require_group(group_name)

        # Step 1: Store datasets
        for key, value in data_dict.items():
            print(f"Saving {key}...")
            # If the dataset already exists and we don’t want to overwrite, skip saving it
            if key in group and not overwrite:
                print(f"Dataset '{key}' already exists in '{group_name}'. Use overwrite=True to replace.")
                continue
            # If we want to overwrite, delete the old dataset
            if key in group and overwrite:
                del group[key]  # Remove existing dataset before overwriting
            # Create the new dataset with compression
            group.create_dataset(key, data=value, compression="gzip")

        # Step 2: Store metadata
        if metadata:
            for meta_key, meta_value in metadata.items():
                print(f"Saving: {meta_key}...")
                print(f'Value: {meta_value}')
                # Convert metadata to a serializable format
                meta_value = convert_to_serializable(meta_value)
                # Store the metadata as a dataset if it's a complex type
                if isinstance(meta_value, (dict, list)):
                    # Serialize the complex metadata into a string for storing as an attribute
                    meta_value = json.dumps(meta_value) 
                # Store the metadata as attributes of the group
                group.attrs[meta_key] = meta_value

        # Step 3: Automatically store timestamp of when the data was saved
        group.attrs["LAST_UPDATED"] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    print(f"Saved data to {file_path} under group '{group_name}' successfully.")

# def save_field_to_file(file_path, group_name, data_dict, metadata=None, overwrite=None):
#     """
#     Saves data to an HDF5 file under a specified group. Allows appending of new datasets, overwriting of existing ones, 
#     and inclusion of metadata with the saved datasets.

#     Args:
#         file_path (str): The path to the HDF5 file where data will be saved.
#         group_name (str): The group name within the HDF5 file to store the data.
#         data_dict (dict): A dictionary where keys are dataset names and values are the corresponding data to store.
#         metadata (dict, optional): A dictionary containing metadata to be stored as attributes in the group. Default is None.
#         overwrite (bool, optional): If True, existing datasets will be overwritten. If False, existing datasets will be skipped. Default is None.

#     Returns:
#         None

#     Raises:
#         None

#     Example:
#         >>> data_dict = {"field1": np.array([1, 2, 3]), "field2": np.array([4, 5, 6])}
#         >>> metadata = {"description": "Sample data", "version": 1.0}
#         >>> save_field_to_file("data.h5", "group1", data_dict, metadata, overwrite=True)
#         Saves the datasets "field1" and "field2" to the "group1" in "data.h5" and includes metadata.

#     Notes:
#         - If the `overwrite` argument is not specified or is None, the function will append the datasets if they don’t already exist.
#         - If the metadata contains non-basic types (e.g., dict, list), they will be converted to JSON strings.
#         - The function stores the timestamp of the save operation as an attribute named "last_updated".
#     """
#     # Default mode is append (this prevents overwriting the file entirely)
#     mode = "a"  # Append mode
#     with h5py.File(file_path, mode) as f:
#         # Get or create the specified group within the HDF5 file
#         group = f.require_group(group_name)

#         # Step 1: Store datasets
#         for key, value in data_dict.items():
#             print(f"Saving {key}...")
#             # If the dataset already exists and we don’t want to overwrite, skip saving it
#             if key in group and not overwrite:
#                 print(f"Dataset '{key}' already exists in '{group_name}'. Use overwrite=True to replace.")
#                 continue
#             # If we want to overwrite, delete the old dataset
#             if key in group and overwrite:
#                 del group[key]  # Remove existing dataset before overwriting
#             # Create the new dataset with compression
#             group.create_dataset(key, data=value, compression="gzip")

#         # Step 2: Store metadata
#         if metadata:
#             for meta_key, meta_value in metadata.items():
#                 print(f"Saving: {meta_key}...")
#                 print(f'Value: {meta_value}')
#                 # Convert metadata to a string if it’s not a basic type
#                 if isinstance(meta_value, (dict, list)):
#                     meta_value = json.dumps(meta_value)  # Convert to JSON string
#                 elif not isinstance(meta_value, (str, int, float, bool, np.number)):
#                     meta_value = str(meta_value)  # Fallback for any unsupported type
#                 # Store the metadata as attributes of the group
#                 group.attrs[meta_key] = meta_value

#         # Step 3: Automatically store timestamp of when the data was saved
#         group.attrs["LAST_UPDATED"] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

#     print(f"Saved data to {file_path} under group '{group_name}' successfully.")

# def load_field_from_file(file_path, group_name):
#     """
#     Load datasets and metadata from an HDF5 file under the specified group.

#     Args:
#         file_path (str): The path to the HDF5 file from which data will be loaded.
#         group_name (str): The group name within the HDF5 file from which datasets and metadata will be loaded.

#     Returns:
#         tuple: A tuple containing two elements:
#             - datasets (dict): A dictionary where keys are dataset names and values are the corresponding data arrays.
#             - metadata (dict): A dictionary where keys are attribute names and values are the corresponding metadata values. Metadata values are automatically converted back from JSON strings if possible.

#     Raises:
#         None

#     Example:
#         >>> datasets, metadata = load_field_from_file("data.h5", "group1")
#         Loads the datasets and metadata from the "group1" in "data.h5".

#     Notes:
#         - If a dataset or attribute is stored as a JSON string, it will be converted back to the original Python type.
#         - If the group name is not found in the file, the function prints an error message and returns None.
#     """
#     # Open the HDF5 file in read mode
#     with h5py.File(file_path, "r") as f:
#         # Check if the specified group exists
#         if group_name not in f:
#             print(f"Group '{group_name}' not found in {file_path}.")
#             return None

#         # Get the specified group
#         group = f[group_name]
        
#         # Step 1: Load datasets into a dictionary
#         datasets = {key: group[key][...] for key in group.keys()}
        
#         # Step 2: Load metadata (convert JSON strings back to Python objects if applicable)
#         metadata = {}
#         for attr in group.attrs.keys():
#             value = group.attrs[attr]
#             if isinstance(value, str):  # Check if the value might be a JSON string
#                 try:
#                     # Attempt to convert JSON strings back to Python objects
#                     metadata[attr] = json.loads(value)
#                 except json.JSONDecodeError:
#                     # Keep as a string if it’s not a valid JSON string
#                     metadata[attr] = value

#     # Return the datasets and metadata
#     return datasets, metadata

def load_field_from_file(file_path, group_name):
    """
    Load datasets and metadata from an HDF5 file under the specified group.

    Args:
        file_path (str): The path to the HDF5 file from which data will be loaded.
        group_name (str): The group name within the HDF5 file from which datasets and metadata will be loaded.

    Returns:
        tuple: A tuple containing two elements:
            - datasets (dict): A dictionary where keys are dataset names and values are the corresponding data arrays.
            - metadata (dict): A dictionary where keys are attribute names and values are the corresponding metadata values. Metadata values are automatically converted back from JSON strings if possible.

    Raises:
        None

    Example:
        >>> datasets, metadata = load_field_from_file("data.h5", "group1")
        Loads the datasets and metadata from the "group1" in "data.h5".

    Notes:
        - If a dataset or attribute is stored as a JSON string, it will be converted back to the original Python type.
        - If the group name is not found in the file, the function prints an error message and returns None.
    """
    # Open the HDF5 file in read mode
    with h5py.File(file_path, "r") as f:
        # Check if the specified group exists
        if group_name not in f:
            print(f"Group '{group_name}' not found in {file_path}.")
            return None

        # Get the specified group
        group = f[group_name]
        
        # Step 1: Load datasets into a dictionary
        datasets = {key: group[key][...] for key in group.keys()}
        
        # Step 2: Load metadata (convert JSON strings back to Python objects if applicable)
        metadata = {}
        
        # First, try to load attributes
        for attr in group.attrs.keys():
            value = group.attrs[attr]
            if isinstance(value, str):  # Check if the value might be a JSON string
                try:
                    # Attempt to convert JSON strings back to Python objects
                    metadata[attr] = json.loads(value)
                except json.JSONDecodeError:
                    # Keep as a string if it’s not a valid JSON string
                    metadata[attr] = value
            else:
                metadata[attr] = value

        # Step 3: Check if metadata was stored as datasets instead of attributes
        for key in group.keys():
            if key not in metadata:  # Only check keys that are not already in metadata (attributes)
                metadata[key] = group[key][...]  # Load the dataset and add it as metadata

    # Return the datasets and metadata
    return datasets, metadata

def print_h5_structure(filepath, n_attributes=5):
    """
    Print the structure of an HDF5 file, including groups, datasets, and the first n attributes.
    Attributes are unpacked if they contain JSON strings and displayed on new lines, only for groups.

    :param filepath: The path to the HDF5 file.
    :param n_attributes: The number of attributes to display per group. Default is 5.
    """
    try:
        # Open the HDF5 file in read-only mode
        with h5py.File(filepath, 'r') as f:
            
            # Helper function to unpack attributes
            def unpack_attributes(group):
                metadata = {}
                for attr in group.attrs.keys():
                    value = group.attrs[attr]
                    if isinstance(value, str):  # Check if the value might be a JSON string
                        try:
                            # Attempt to convert JSON strings back to Python objects
                            metadata[attr] = json.loads(value)
                        except json.JSONDecodeError:
                            # Keep as a string if it's not a valid JSON string
                            metadata[attr] = value
                    else:
                        metadata[attr] = value
                return metadata
            
            # Function to print group and dataset structure
            def print_group(name, obj):
                print("\n" + "-"*40)  # Separator for better readability
                print(f"ITEM: {name}")
                print("-" * 40)
                
                if isinstance(obj, h5py.Group):
                    # It's a group, print its name and info
                    print(f"Group: {name}")
                    print(f"  Type: Group")
                    print(f"  Number of Datasets: {len(obj)}")
                    # Print the first n attributes (on new lines)
                    attributes = unpack_attributes(obj)
                    if attributes:
                        print(f"  Attributes (first {n_attributes}):")
                        for idx, (attr_name, attr_value) in enumerate(list(attributes.items())[:n_attributes]):
                            print(f"    {attr_name}: {attr_value}")
                    else:
                        print(f"  No attributes for this group.")
                    
                    # Print datasets
                    for key, item in obj.items():
                        if isinstance(item, h5py.Dataset):
                            print(f"  Dataset: {key}")
                            print(f"    No attributes for this dataset.")

                elif isinstance(obj, h5py.Dataset):
                    # It's a dataset, print its name and info (no attributes section)
                    print(f"Dataset: {name}")
                    print(f"  Type: Dataset")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")
                    print(f"  Number of Elements: {obj.size}")

                print("-" * 40)  # End separator for clarity
            
            # Visit all items in the HDF5 file recursively
            f.visititems(print_group)

    except Exception as e:
        print(f"Error opening file: {e}")

def load_selected_datasets_from_file(file_path, group_name, dataset_names):
    """
    Load specified datasets and their metadata from an HDF5 file under the given group.

    Args:
        file_path (str): The path to the HDF5 file from which data will be loaded.
        group_name (str): The group name within the HDF5 file from which datasets and metadata will be loaded.
        dataset_names (str or list): A dataset name (or list of names) to be loaded from the group.

    Returns:
        tuple: A tuple containing two elements:
            - datasets (dict): A dictionary where keys are dataset names and values are the corresponding data arrays.
            - metadata (dict): A dictionary where keys are attribute names and values are the corresponding metadata values.
                Metadata is loaded for the specified datasets.

    Raises:
        ValueError: If any of the requested datasets are not found in the group.

    Example:
        >>> datasets, metadata = load_selected_datasets_from_file("data.h5", "group1", ["dataset1", "dataset2"])
        Loads the datasets "dataset1" and "dataset2" from the "group1" in "data.h5".

    Notes:
        - If a dataset or attribute is stored as a JSON string, it will be converted back to the original Python type.
    """
    # Ensure dataset_names is a list (if a single dataset name is passed as a string)
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Open the HDF5 file in read mode
    with h5py.File(file_path, "r") as f:
        # Check if the specified group exists
        if group_name not in f:
            print(f"Group '{group_name}' not found in {file_path}.")
            return None

        # Get the specified group
        group = f[group_name]
        
        # Initialize dictionaries to hold the datasets and metadata
        datasets = {}
        metadata = {}

        # Step 1: Load selected datasets into the datasets dictionary
        for name in dataset_names:
            if name in group:
                datasets[name] = group[name][...]  # Load the dataset
            else:
                print(f"Dataset '{name}' not found in group '{group_name}'.")
                raise ValueError(f"Dataset '{name}' not found in group '{group_name}'.")

        # Step 2: Load metadata only for the requested datasets
        for name in dataset_names:
            if name in group.attrs:
                value = group.attrs[name]
                if isinstance(value, str):  # Check if the value might be a JSON string
                    try:
                        # Attempt to convert JSON strings back to Python objects
                        metadata[name] = json.loads(value)
                    except json.JSONDecodeError:
                        # Keep as a string if it’s not a valid JSON string
                        metadata[name] = value
                else:
                    metadata[name] = value

    # Return the selected datasets and their metadata
    return datasets, metadata