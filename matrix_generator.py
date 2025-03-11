"""
Code generating Ewald and Buckingham interaction matrices
Phase object works only with core model of Buckingham potential
meaning that the parser is implemented

Garnet phase field Ca3Al2Si3O12 is hardcoded and no parser is available
it kicks in as soon as the phase field directory has no file with
a potential

Multiprocessing and numba was used to accelerate the computations
"""

# this file is full of file path hacks, fix it at some point

import numpy as np
import os

from ipcsp import root_dir
from ipcsp.grids_and_symmetry import cubic
import math
from numba import jit, njit
from time import sleep, time
from multiprocessing import Pool
from functools import partial

filedir = root_dir / 'data/'


class Phase:
    filedir = root_dir / 'data/'

    def __init__(self, phase_name, use_ml_potential=False, ml_model_name="M3GNet-MP-2021.2.8-PES"):
        """
        Initialize a Phase object with crystal parameters.
        
        Args:
            phase_name (str): Name of the phase directory in data/
            use_ml_potential (bool): Whether to use ML potentials for short-range interactions
            ml_model_name (str): Name of the MatGL model to use for ML potentials
        """
        self.types = []

        self.radius = {}

        self.closest_distance = {}  # dict['O', 'O'] = 2.2 A

        self.charge = {}

        self.buck = {}
        
        # ML potential settings
        self.use_ml_potential = use_ml_potential
        self.ml_model_name = ml_model_name
        self.ml_model = None  # Will be initialized when needed

        self.name = phase_name

        # Use absolute paths to avoid issues with current working directory
        self.location = root_dir / 'data' / phase_name

        self.garnet = False  # HACK:garnet case is treated separately

        # Use os.path.join for path construction and make sure the path is correct
        radii_path = os.path.join(root_dir, 'data', phase_name, 'radii.lib')
        try:
            with open(radii_path, 'r') as f:
                for line in f.readlines():
                    if line.startswith('#'):
                        continue
                    line = line.rstrip('\n')
                    line = line.split()
                    self.types.append(line[0])
                    self.radius[line[0]] = float(line[1])
            print('Radii', self.radius)
        except FileNotFoundError:
            print(f"Error: Could not find radii.lib file at {radii_path}")
            print(f"Current root_dir is {root_dir}")
            raise

        try:
            dist_path = os.path.join(root_dir, 'data', phase_name, 'dist.lib')
            with open(dist_path, 'r') as f:
                for line in f.readlines():
                    if line.startswith('#'):
                        continue
                    line = line.rstrip('\n')
                    line = line.split()
                    pair = (min(line[0], line[1]), max(line[0], line[1]))
                    self.closest_distance[pair] = float(line[2])
            print('Overriding Shannon radii with the following closest distances:', self.closest_distance)
        except IOError:
            print("No closest distances between the ions were provided")
            print("Relying on Shannon radii only")

        try:
            buck_path = os.path.join(root_dir, 'data', phase_name, 'buck.lib')
            with open(buck_path, 'r') as f:
                charge_lines = False
                buck_lines = False

                for line in f.readlines():
                    if line.startswith('#'):
                        continue
                    line = line.rstrip('\n')
                    if len(line) > 0:

                        if 'species' in line:
                            charge_lines = True
                            buck_lines = False
                            continue

                        if 'buck' in line:
                            buck_lines = True
                            charge_lines = False
                            continue

                        if charge_lines:
                            line = line.split()
                            self.charge[line[0]] = float(line[-1])

                        if buck_lines:
                            line = line.split()
                            pair = (min(line[0], line[2]), max(line[0], line[2]))
                            self.buck[pair] = {}
                            self.buck[pair]['par'] = list(map(float, line[4:7]))
                            self.buck[pair]['lo'] = float(line[7])
                            self.buck[pair]['hi'] = float(line[-1])              
        except IOError:
            print("There is no buck file! I assume that we are dealing with the garnet problem.")

            # Handcrafted parameters, write a parser later on
            self.gar_param = {}
            self.gar_param[('Al', 'O')] = {'De': 0.361581, 'a0': 1.900442, 'r0': 2.164818, 'A': 0.9, 'lo': 0.0,
                                           'hi': 15.0}
            self.gar_param[('Ca', 'O')] = {'De': 0.030211, 'a0': 2.2413340, 'r0': 2.923245, 'A': 5.0, 'lo': 0.0,
                                           'hi': 15.0}
            self.gar_param[('O', 'O')] = {'De': 0.042395, 'a0': 1.379316, 'r0': 3.618701, 'A': 22.0, 'lo': 0.0,
                                          'hi': 15.0}
            self.gar_param[('O', 'Si')] = {'De': 0.340554, 'a0': 2.0067, 'r0': 2.1, 'A': 1.0, 'lo': 0.0, 'hi': 15.0}

            self.charge['Al'] = 1.8
            self.charge['Ca'] = 1.2
            self.charge['O'] = -1.2
            self.charge['Si'] = 2.4

            self.garnet = True

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


@njit(cache=True)
def QEwald(positions, vecs, reciprocal, cell_volume, alpha=-1):
    """
    positions are relative coordinates
    vecs are the lattice vectors
    reciprocal are the reciprocal lattice vectors. I need them only to use numba, since it doesn't support cross product
    cell_volume is the volume of the cell
    alpha controls the split between the direct and reciprocal sums

    You can tweak realDepth and reciprocalDepth summation constant below in the code
    """
    N = len(positions)
    if alpha < 0:
        alpha = 2 / (cell_volume ** (1.0 / 3))
    realDepth = 4  # 3
    reciprocalDepth = 4  # 3

    pos = positions @ vecs

    d = np.zeros((N, N))

    shifts = np.zeros(((2 * realDepth + 1) ** 3 - 1, 3))
    i = 0
    tmp = np.array([realDepth, realDepth, realDepth])

    for shift in np.ndindex(2 * realDepth + 1, 2 * realDepth + 1, 2 * realDepth + 1):
        if shift != (realDepth, realDepth, realDepth):
            shifts[i,] = shift
            shifts[i,] = shifts[i,] - tmp
            i = i + 1

    shifts = shifts @ vecs

    for i in np.arange(N):
        for j in np.arange(i, N):
            if i != j:
                r = np.linalg.norm(pos[i,] - pos[j,])
                d[i, j] += math.erfc(alpha * r) / (2 * r)

            for s in np.arange(len(shifts)):
                r = np.linalg.norm(pos[i,] + shifts[s,] - pos[j,])
                d[i, j] += math.erfc(alpha * r) / (2 * r)

    # self interaction term
    for i in np.arange(N):
        d[i, i] = d[i, i] - alpha / math.sqrt(math.pi)

    # Ewald reciprocal space

    shifts_recip = np.zeros(((2 * reciprocalDepth + 1) ** 3 - 1, 3))
    i = 0
    tmp = np.array([reciprocalDepth, reciprocalDepth, reciprocalDepth])

    for shift_recip in np.ndindex(2 * reciprocalDepth + 1, 2 * reciprocalDepth + 1, 2 * reciprocalDepth + 1):
        if shift_recip != (reciprocalDepth, reciprocalDepth, reciprocalDepth):
            shifts_recip[i,] = shift_recip
            shifts_recip[i,] = shifts_recip[i,] - tmp
            i = i + 1

    shifts_recip = shifts_recip @ reciprocal

    for i in np.arange(N):
        for j in np.arange(i, N):

            for s in np.arange(len(shifts_recip)):
                k = shifts_recip[s,]
                # k = np.array(shift)@self.reciprocal
                term = (4 * math.pi ** 2) / np.dot(k, k)
                term = term * math.exp(-np.dot(k, k) / (4 * alpha ** 2))
                v = pos[j,] - pos[i,]
                term = term * math.cos(np.dot(k, v))
                d[i, j] += term / (2 * math.pi * cell_volume)

    # Unit conversion
    d = d * 14.399645351950543

    # symmetry completion
    for i in np.arange(N):
        for j in np.arange(i):
            d[i, j] = d[j, i]

    return d


def generate_Ewald(size, ouput_directory):
    """
    Start with the cubic systems
    """
    filename = f'C{size}.npy'
    
    # Make sure the output directory exists
    os.makedirs(ouput_directory, exist_ok=True)
    
    # Generate the grid and compute preliminary parameters
    positions = cubic(size)

    vecs = np.zeros((3, 3))
    for i in range(3):
        vecs[i, i] = size
    cell_volume = abs(np.linalg.det(vecs))

    reciprocal = np.zeros((3, 3))

    for i in np.arange(3):
        recip_vector = 2 * math.pi * np.cross(vecs[(1 + i) % 3,], vecs[(2 + i) % 3]) / cell_volume
        reciprocal[i,] = recip_vector

    dist = QEwald(positions, vecs, reciprocal, cell_volume)
    print('Ewald matrix for cubic system of size {size} was generated'.format(size=size))
    print("Its max is ", dist.max())
    print("Its min is ", dist.min())
    # print(dist)

    output_file_path = os.path.join(ouput_directory, filename)
    with open(output_file_path, 'wb') as outfile:
        np.save(outfile, dist)
    
    print(f"Saved Ewald matrix to {output_file_path}")


@njit(cache=True)
def BuckinghamTwoIons(pos_i, pos_j, cell_size, A, rho, beta, lo, hi, closest_distance):
    """
    Interaction between two ions under PBC
    pos_i, pos_j belongs to 3D

    TODO:Double check the ranges!
    """
    MAX = 300
    max_cells = int(np.ceil(hi / cell_size))
    energy = 0

    # HACK BEGINS
    # Dirty hack to handle closest distances
    # TODO: remove this
    if rho < 0.00001:
        max_cells = 1

        for i in range(-max_cells, max_cells + 1):
            for j in range(-max_cells, max_cells + 1):
                for k in range(-max_cells, max_cells + 1):
                    if not (i == 0 and j == 0 and k == 0):
                        r = cell_size * np.linalg.norm(pos_j + np.array([i, 0, 0]) +
                                                       np.array([0, j, 0]) + np.array([0, 0, k]) - pos_i)

                        if r < closest_distance:
                            return MAX

        if np.linalg.norm(pos_i - pos_j) > 0.001:  # interaciton within the cell, you can remove it to the row function
            r = cell_size * np.linalg.norm(pos_j - pos_i)

            if r < closest_distance:
                return MAX

        return 0
    # HACK ENDS
    else:
        # interaction with neighbouring cells
        for i in range(-max_cells, max_cells + 1):
            for j in range(-max_cells, max_cells + 1):
                for k in range(-max_cells, max_cells + 1):
                    if not (i == 0 and j == 0 and k == 0):
                        r = cell_size * np.linalg.norm(pos_j + np.array([i, 0, 0]) +
                                                       np.array([0, j, 0]) + np.array([0, 0, k]) - pos_i)
                        if r <= hi:
                            energy += A * math.exp(-1.0 * r / rho) - beta / r ** 6

                        if r < closest_distance:
                            return MAX

        if np.linalg.norm(
                pos_i - pos_j) > 0.001:  # add interaciton within the cell, you can remove it to the row function
            r = cell_size * np.linalg.norm(pos_j - pos_i)
            if r <= hi:
                energy += A * math.exp(-1.0 * r / rho) - beta / r ** 6

            if r < closest_distance:
                return MAX

    return energy


@njit(cache=True)
def GarnetTwoIons(pos_i, pos_j, cell_size, De, a0, r0, A, lo, hi, closest_distance):
    """
    Interaction between two ions under PBC
    pos_i, pos_j belongs to 3D
    """
    MAX = 5000
    max_cells = int(np.ceil(hi / cell_size))
    energy = 0

    # interaction with neighbouring cells
    for i in range(-max_cells, max_cells + 1):
        for j in range(-max_cells, max_cells + 1):
            for k in range(-max_cells, max_cells + 1):
                if not (i == 0 and j == 0 and k == 0):
                    r = cell_size * np.linalg.norm(pos_j + np.array([i, 0, 0]) +
                                                   np.array([0, j, 0]) + np.array([0, 0, k]) - pos_i)
                    if r <= hi:
                        energy += De * ((1 - math.exp(-a0 * (r - r0))) ** 2 - 1) + (A / r ** 12)

                    if r < closest_distance:
                        return MAX

    if np.linalg.norm(pos_i - pos_j) > 0.001:  # add interaciton within the cell, you can remove it to the row function
        r = cell_size * np.linalg.norm(pos_j - pos_i)
        if r <= hi:
            energy += De * ((1 - math.exp(-a0 * (r - r0))) ** 2 - 1) + (A / r ** 12)

        if r < closest_distance:
            return MAX

    return energy


def _Buck_row(pos_i, grid_size, cell_size, A, rho, beta, lo, hi, closest_distance):
    """
    elements is the pair of elements for which we are going to compute the matrix
    """
    result = np.zeros(grid_size ** 3)

    positions = cubic(grid_size)
    for pos_j in range(pos_i, grid_size ** 3):
        # print(BuckinghamTwoIons(pos_i, pos_j, cell_size, A, rho, beta, lo, hi, radius_threshold))
        result[pos_j] = BuckinghamTwoIons(positions[pos_i], positions[pos_j], cell_size, A, rho, beta, lo, hi,
                                          closest_distance)
    return result


def _gar_row(pos_i, grid_size, cell_size, De, a0, r0, A, lo, hi, closest_distance):
    """
    elements is the pair of elements for which we are going to compute the matrix
    """
    result = np.zeros(grid_size ** 3)

    positions = cubic(grid_size)
    for pos_j in range(pos_i, grid_size ** 3):
        result[pos_j] = GarnetTwoIons(positions[pos_i], positions[pos_j], cell_size, De, a0, r0, A, lo, hi,
                                      closest_distance)
    return result


def generate_Buck(grid_size, cell_size, phase, output_directory, radius_threshold=0.75, multicpu=False):
    """
    Cubic systems only for the moment
    size is the cube size
    phase
    TODO: compute distances once, e.g. (i1, i2): [cell (0,0,0), cell (1,0,0), cell (0,1,0), cell (0,0,1), ...]
    Then apply the Buckingham function
    """
    for ion_pair, potential_param in phase.buck.items():

        result = np.zeros((grid_size ** 3, grid_size ** 3))
        closest_distance = radius_threshold * (phase.radius[ion_pair[0]] + phase.radius[ion_pair[1]])

        if ion_pair in phase.closest_distance:
            closest_distance = phase.closest_distance[ion_pair]
            print("Closest distance for " + ion_pair[0] + '-' + ion_pair[1] + " was set to be " + str(
                phase.closest_distance[ion_pair]))

        buck_row = partial(_Buck_row, grid_size=grid_size, cell_size=cell_size, A=potential_param['par'][0],
                           rho=potential_param['par'][1], beta=potential_param['par'][2],
                           lo=potential_param['lo'], hi=potential_param['hi'], closest_distance=closest_distance)

        if multicpu:
            with Pool(processes=4) as pool:
                i = 0
                for row in pool.map(buck_row, range(grid_size ** 3)):
                    result[i,] = row
                    i += 1
            # pass
        else:
            i = 0
            for row in map(buck_row, range(grid_size ** 3)):
                result[i,] = row
                i += 1

        # for i in prange(size):
        #     result[i] = test()

        filename = 'C{grid_size}_'.format(grid_size=grid_size) + ion_pair[0] + ion_pair[1] + '_{cell_size}.npy'.format(
            cell_size=cell_size)

        # print(filename)
        # print(ion_pair, '\n', np.around(result, decimals=2))

        with open(output_directory / filename, 'wb') as outfile:
            np.save(outfile, result)


def generate_garnet(grid_size, cell_size, phase, output_directory, radius_threshold=0.75, multicpu=False):
    """
    HACK for the garnet
    """
    for ion_pair, potential_param in phase.gar_param.items():

        result = np.zeros((grid_size ** 3, grid_size ** 3))
        closest_distance = radius_threshold * (phase.radius[ion_pair[0]] + phase.radius[ion_pair[1]])

        if phase.closest_distance is not None and ion_pair in phase.closest_distance:
            closest_distance = phase.closest_distance[ion_pair]
            print("Closest distance for " + ion_pair[0] + '-' + ion_pair[1] + " was set to be " + str(
                phase.closest_distance[ion_pair]))

        # def GarnetTwoIons(pos_i, pos_j, cell_size, De, a0, r0, A, lo, hi, closest_distance):
        garnet_row = partial(_gar_row, grid_size=grid_size, cell_size=cell_size, De=potential_param['De'],
                             a0=potential_param['a0'], r0=potential_param['r0'],
                             A=potential_param['A'], lo=potential_param['lo'], hi=potential_param['hi'],
                             closest_distance=closest_distance)

        if multicpu:
            with Pool(processes=4) as pool:
                i = 0
                for row in pool.map(garnet_row, range(grid_size ** 3)):
                    result[i,] = row
                    i += 1
            # pass
        else:
            i = 0
            for row in map(garnet_row, range(grid_size ** 3)):
                result[i,] = row
                i += 1

        filename = 'C{grid_size}_'.format(grid_size=grid_size) + ion_pair[0] + ion_pair[1] + '_{cell_size}.npy'.format(
            cell_size=cell_size)

        with open(output_directory / filename, 'wb') as outfile:
            np.save(outfile, result)


def get_Buck(ion_pair, grid_size, cell_size, phase):
    """
    Loads or generates buckingham matrix
    """
    filename = 'C{grid_size}_'.format(grid_size=grid_size) + ion_pair[0] + ion_pair[1] + '_{cell_size}.npy'.format(
        cell_size=cell_size)

    try:
        with open(phase.location / filename, 'rb') as f:
            return np.load(f)
    except IOError:
        generate_Buck(grid_size, cell_size, phase, phase.location, multicpu=True)
        with open(phase.location / filename, 'rb') as f:
            return np.load(f)


def get_garnet(ion_pair, grid_size, cell_size, phase):
    """
    HACK:garnet
    """
    filename = 'C{grid_size}_'.format(grid_size=grid_size) + ion_pair[0] + ion_pair[1] + '_{cell_size}.npy'.format(
        cell_size=cell_size)

    try:
        with open(phase.location / filename, 'rb') as f:
            return np.load(f)
    except IOError:
        generate_garnet(grid_size, cell_size, phase, phase.location, multicpu=True)
        with open(phase.location / filename, 'rb') as f:
            return np.load(f)


def get_Ewald(grid_size, cell_size):
    """
    Function was rewritten since my old trick with float didn't work
    """
    filename = f'C{grid_size}.npy'
    
    # Create Ewald directory path
    ewald_dir = os.path.join(root_dir, 'data', 'Ewald')
    os.makedirs(ewald_dir, exist_ok=True)
    
    # Full path to the Ewald file
    ewald_file_path = os.path.join(ewald_dir, filename)
    
    try:
        with open(ewald_file_path, 'rb') as f:
            print(f"Loading Ewald matrix from {ewald_file_path}")
            return np.load(f)* (grid_size / cell_size)
    except FileNotFoundError:
        print(f"Ewald matrix file not found at {ewald_file_path}, generating...")
        # Generate the Ewald matrix if it doesn't exist
        generate_Ewald(grid_size, ewald_dir)
        # Try loading again
        with open(ewald_file_path, 'rb') as f:
            return np.load(f)* (grid_size / cell_size)


def generate_ML_potential(grid_size, cell_size, phase, output_directory, radius_threshold=0.75):
    """
    Generate energy matrices using ML potentials for all ion pairs.
    
    Args:
        grid_size (int): Size of the grid
        cell_size (float): Size of the cell in Angstroms
        phase (Phase): Phase object with ion information
        output_directory (Path): Directory to save output matrices
        radius_threshold (float): Threshold for minimum distance as fraction of sum of ionic radii
        
    Returns:
        dict: Dictionary of energy matrices for each ion pair
    """
    from ipcsp.ml_potential import MLPotential
    
    print(f"Generating ML potential matrices for grid size {grid_size}")
    
    # Initialize ML potential if not already done
    if phase.ml_model is None:
        phase.ml_model = MLPotential(model_name=phase.ml_model_name)
    
    # Generate energy matrices for all ion pairs
    types = phase.types
    energy_matrices = {}
    
    for i, ion1 in enumerate(types):
        for j, ion2 in enumerate(types[i:], i):
            ion_pair = (min(ion1, ion2), max(ion1, ion2))
            print(f"Calculating ML potential for {ion_pair}...")
            
            # Check if the file already exists
            filename = f"ML_{ion_pair[0]}_{ion_pair[1]}_G{grid_size}.npy"
            
            if os.path.exists(output_directory / filename):
                print(f"ML potential for {ion_pair} already exists. Loading from file...")
                with open(output_directory / filename, 'rb') as infile:
                    energy_matrices[ion_pair] = np.load(infile)
                continue
            
            # Calculate the energy matrix for this ion pair
            energy_matrix = phase.ml_model.calculate_energy_matrix(grid_size, cell_size, phase)
            energy_matrices[ion_pair] = energy_matrix
            
            # Save the matrix to file
            with open(output_directory / filename, 'wb') as outfile:
                np.save(outfile, energy_matrix)
                
    return energy_matrices

def get_ML_potential(ion_pair, grid_size, cell_size, phase):
    """
    Get the ML potential energy matrix for a specific ion pair.
    
    Args:
        ion_pair (tuple): Tuple of ion symbols (str, str)
        grid_size (int): Size of the grid
        cell_size (float): Size of the cell in Angstroms
        phase (Phase): Phase object with ion information
        
    Returns:
        np.ndarray: Energy matrix for the ion pair
    """
    # First try to load from file
    filename = f"ML_{ion_pair[0]}_{ion_pair[1]}_G{grid_size}.npy"
    try:
        with open(os.path.join(phase.location, filename), 'rb') as infile:
            return np.load(infile)
    except:
        # If file doesn't exist, calculate on the fly
        from ipcsp.ml_potential import MLPotential
        
        if phase.ml_model is None:
            phase.ml_model = MLPotential(model_name=phase.ml_model_name)
            
        print(f"Calculating ML potential for {ion_pair} on the fly...")
        return phase.ml_model.calculate_energy_matrix(grid_size, cell_size, phase)


if __name__ == "__main__":
    # Code to generate Ewald matrices
    # size_range = range(17,19)
    # # size_range = range(2,3)
    # output_directory = filedir+'Ewald/'
    # for size in size_range:
    #     generate_Ewald(size, ouput_directory)

    # Code to generate Buck matrices
    SrTiO = Phase('SrTiO')
    print("The following parameters for the unit cell are used:", SrTiO)
    output_directory = filedir + 'SrTiO/'
    # grid_range = range(2, 4)
    grid_range = range(5, 9)
    for grid in grid_range:
        start = time()
        generate_Buck(grid, 3.9, SrTiO, output_directory, multicpu=True)
        end = time()
        print('Grid size {grid} took '.format(grid=grid), end='')
        print(" %s seconds" % (end - start))

    # Testing the timing
    # start = time()
    # generate_Buck(10)
    # end = time()
    # print("Time for parallel = %s" % (end - start))
    # generate_Buck(10)
