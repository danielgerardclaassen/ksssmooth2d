from skfem import *
from skfem.helpers import dot, grad
import numpy as np

# Default number of nodes in each spatial direction (x, y)
DEFAULT_NODECOUNTS_2D = (16, 16)
DEFAULT_NODECOUNT_1D = 32

# Default finite element type (Linear triangular elements)
DEFAULT_ELEMENT_2D = ElementTriP1()

def setup_2d_fem(bc_vals, nodecounts=DEFAULT_NODECOUNTS_2D, element=DEFAULT_ELEMENT_2D):
    """
    Sets up a 2D finite element problem on a triangular mesh.

    Parameters:
    -----------
    bc_vals : tuple
        Boundary values (x_min, x_max, y_min, y_max) defining the domain.
    nodecounts : tuple, optional
        Number of nodes in the x and y directions, respectively (default: (16,16)).
    element : skfem Element, optional
        The finite element type to use (default: linear triangular elements).

    Returns:
    --------
    coordinates : np.ndarray
        2D array of x and y coordinates of the nodes.
    mesh : skfem MeshTri
        The triangular mesh generated from the node grid.
    basis : skfem Basis
        Finite element basis functions associated with the mesh.
    basis_coordinates : np.ndarray
        Array of degrees of freedom (DOF) locations for the basis.
    """
    # Generate evenly spaced nodes along x and y directions
    x_coordinates = np.linspace(bc_vals[0], bc_vals[1], nodecounts[0])
    y_coordinates = np.linspace(bc_vals[2], bc_vals[3], nodecounts[1])
    
    # Combine x and y coordinates into a single array (not strictly necessary)
    coordinates = np.asarray((x_coordinates, y_coordinates))

    # Create a triangular mesh from the tensor grid of x and y coordinates
    mesh = MeshTri.init_tensor(x_coordinates, y_coordinates)

    # Define basis functions on the mesh using the specified finite element
    basis = Basis(mesh, element)

    # Extract the DOF locations in a transposed format (each row is a coordinate pair)
    basis_coordinates = basis.doflocs.T

    return coordinates, mesh, basis, basis_coordinates

def setup_1d_fem_p2p1(bc_vals, nodecounts=DEFAULT_NODECOUNTS_2D):
    """
    Sets up a 1D finite element problem on a triangular mesh.

    Parameters:
    -----------
    bc_vals : tuple
        Boundary values (x_min, x_max, y_min, y_max) defining the domain.
    nodecounts : tuple, optional
        Number of nodes in the x and y directions, respectively (default: (16,16)).
    element : skfem Element, optional
        The finite element type to use (default: linear triangular elements).

    Returns:
    --------
    coordinates : np.ndarray
        2D array of x and y coordinates of the nodes.
    mesh : skfem MeshTri
        The triangular mesh generated from the node grid.
    basis : skfem Basis
        Finite element basis functions associated with the mesh.
    basis_coordinates : np.ndarray
        Array of degrees of freedom (DOF) locations for the basis.
    """
    # Generate evenly spaced nodes along x and y directions
    x_coordinates = np.linspace(bc_vals[0], bc_vals[1], nodecounts[0])

    # Create a triangular mesh from the tensor grid of x and y coordinates
    mesh = MeshLine(x_coordinates)

    boundary = {
        "wall": mesh.facets_satisfying(lambda x: np.logical_or(x[0] == bc_vals[0], x[0] == bc_vals[1]))
    }

    # Define basis functions on the mesh using the specified finite element
    u_basis = InteriorBasis(mesh, ElementLineP2())
    eta_basis = u_basis.with_element(ElementLineP1())

    bases = {
        "u" : u_basis,
        "eta": eta_basis,
        **{
        label: FacetBasis(mesh, ElementLineP2(), facets = boundary[label]) 
        for label in ['wall']
        }

    }

    # Extract the DOF locations in a transposed format (each row is a coordinate pair)
    u_basis_coordinates = bases["u"].doflocs.T
    eta_basis_coordinates = bases["eta"].doflocs.T

    basis_coordinates = {
        "u": u_basis_coordinates,
        "eta": eta_basis_coordinates
    }

    return x_coordinates, mesh, bases, basis_coordinates

def setup_2d_fem_p2p1(bc_vals, nodecounts=DEFAULT_NODECOUNTS_2D):
    """
    Sets up a 2D finite element problem on a triangular mesh with P2P1 elements.

    Parameters:
    -----------
    bc_vals : tuple
        Boundary values (x_min, x_max, y_min, y_max) defining the domain.
    nodecounts : tuple, optional
        Number of nodes in the x and y directions, respectively (default: (16,16)).
    element : skfem Element, optional
        The finite element type to use (default: linear triangular elements).

    Returns:
    --------
    coordinates : np.ndarray
        2D array of x and y coordinates of the nodes.
    mesh : skfem MeshTri
        The triangular mesh generated from the node grid.
    basis : skfem Basis
        Finite element basis functions associated with the mesh.
    basis_coordinates : np.ndarray
        Array of degrees of freedom (DOF) locations for the basis.
    """
    # Generate evenly spaced nodes along x and y directions
    x_coordinates = np.linspace(bc_vals[0], bc_vals[1], nodecounts[0])
    y_coordinates = np.linspace(bc_vals[2], bc_vals[3], nodecounts[1])
    
    # Combine x and y coordinates into a single array (not strictly necessary)
    coordinates = np.asarray((x_coordinates, y_coordinates))

    # Create a triangular mesh from the tensor grid of x and y coordinates
    mesh = MeshTri.init_tensor(x_coordinates, y_coordinates)

    # Define basis functions on the mesh using the specified finite element
    u_basis = Basis(mesh, ElementTriP2(), intorder=3)
    eta_basis = Basis(mesh, ElementTriP1(), intorder=3)

    bases = {
        "u" : u_basis,
        "eta": eta_basis
    }

    # Extract the DOF locations in a transposed format (each row is a coordinate pair)
    u_basis_coordinates = bases["u"].doflocs.T
    eta_basis_coordinates = bases["eta"].doflocs.T

    basis_coordinates = {
        "u": u_basis_coordinates,
        "eta": eta_basis_coordinates
    }

    return coordinates, mesh, bases, basis_coordinates

def setup_1d_fem(bc_vals, nodecount=DEFAULT_NODECOUNT_1D):
    """
    Sets up a 1D finite element problem on a triangular mesh.

    Parameters:
    -----------
    bc_vals : tuple
        Boundary values (x_min, x_max) defining the domain.
    nodecounts : int, optional
        Number of nodes in the x direction (default: 32).

    Returns:
    --------
    coordinates : np.ndarray
        1D array of x coordinates of the nodes.
    mesh : skfem MeshTri
        The triangular mesh generated from the node grid.
    basis : skfem Basis
        Finite element basis functions associated with the mesh.
    basis_coordinates : np.ndarray
        Array of degrees of freedom (DOF) locations for the basis.
    """
    # Generate evenly spaced nodes along x and y directions
    x_coordinates = np.linspace(bc_vals[0], bc_vals[1], nodecount)

    # Create a triangular mesh from the tensor grid of x and y coordinates
    mesh = MeshLine(x_coordinates)

    boundary = {
        "wall": mesh.facets_satisfying(lambda x: np.logical_or(x[0] == bc_vals[0], x[0] == bc_vals[1]))
    }

    # Define basis functions on the mesh using the specified finite element
    u_basis = InteriorBasis(mesh, ElementLineP1())
    eta_basis = u_basis.with_element(ElementLineP1())

    bases = {
        "u" : u_basis,
        "eta": eta_basis,
        **{
        label: FacetBasis(mesh, ElementLineP2(), facets = boundary[label]) 
        for label in ['wall']
        }

    }

    # Extract the DOF locations in a transposed format (each row is a coordinate pair)
    u_basis_coordinates = bases["u"].doflocs.T

    return x_coordinates, mesh, u_basis, u_basis_coordinates