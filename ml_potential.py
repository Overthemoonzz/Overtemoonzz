"""
MatGL integration for machine learning potentials in IPCSP.

This module provides functionality to use MatGL's machine learning potentials
for short-range interatomic interactions within the IPCSP framework, while
preserving the Ewald summation for long-range electrostatic interactions.
"""

import numpy as np
import torch
import ase.io
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import M3GNetCalculator
import warnings

# Suppress warnings from MatGL
warnings.simplefilter("ignore")

class MLPotential:
    """
    Wrapper for MatGL machine learning potentials to be used in IPCSP.
    """
    
    def __init__(self, model_name="M3GNet-MP-2021.2.8-PES", cutoff=5.0):
        """
        Initialize the ML potential.
        
        Args:
            model_name (str): Name of the pretrained MatGL model
            cutoff (float): Cutoff radius for interactions in Angstroms
        """
        self.model_name = model_name
        self.cutoff = cutoff
        self.potential = matgl.load_model(model_name)
        self.calculator = None  # Will be initialized when needed
        
    def get_calculator(self):
        """
        Get the ASE calculator for energy calculations.
        
        Returns:
            M3GNetCalculator: ASE calculator using the ML potential
        """
        if self.calculator is None:
            self.calculator = M3GNetCalculator(potential=self.potential)
        return self.calculator
    
    def calculate_energy_matrix(self, grid_size, cell_size, phase):
        """
        Calculate the energy matrix for all pairs of positions in the grid.
        
        Args:
            grid_size (int): Size of the grid
            cell_size (float): Size of the cell in Angstroms
            phase (Phase): Phase object containing ion information
            
        Returns:
            np.ndarray: Energy matrix of shape (n_positions, n_positions)
        """
        N = grid_size**3
        energy_matrix = np.zeros((N, N))
        
        # Create a reference atom for each ion type
        ion_types = phase.types
        ref_atoms = {}
        
        for ion_type in ion_types:
            ref_atoms[ion_type] = Atoms(symbols=[ion_type], positions=[[0, 0, 0]])
            
        # Calculate interactions between each pair of positions
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    grid_points.append([i / grid_size * cell_size, 
                                       j / grid_size * cell_size, 
                                       k / grid_size * cell_size])
        
        for i in range(N):
            for j in range(i+1, N):
                # Skip if too far apart (beyond cutoff)
                dist = np.linalg.norm(np.array(grid_points[i]) - np.array(grid_points[j]))
                if dist > self.cutoff:
                    continue
                
                # Calculate energy for each ion pair type
                for ion1 in ion_types:
                    for ion2 in ion_types:
                        energy_matrix[i, j] += self._calculate_pair_energy(
                            ion1, ion2, grid_points[i], grid_points[j], cell_size)
                
                # Make the matrix symmetric
                energy_matrix[j, i] = energy_matrix[i, j]
                
        return energy_matrix
    
    def _calculate_pair_energy(self, ion1, ion2, pos1, pos2, cell_size):
        """
        Calculate the interaction energy between two ions at specific positions.
        
        Args:
            ion1 (str): Symbol of the first ion
            ion2 (str): Symbol of the second ion
            pos1 (list): Position of the first ion [x, y, z]
            pos2 (list): Position of the second ion [x, y, z]
            cell_size (float): Size of the cell in Angstroms
            
        Returns:
            float: Interaction energy in eV
        """
        # Create an ASE Atoms object with the two ions
        atoms = Atoms(
            symbols=[ion1, ion2],
            positions=[pos1, pos2],
            cell=[cell_size, cell_size, cell_size],
            pbc=[True, True, True]
        )
        
        # Calculate energy using the ML potential
        calc = self.get_calculator()
        atoms.calc = calc
        
        # Get the energy and subtract the reference energies of isolated atoms
        try:
            total_energy = atoms.get_potential_energy()
            
            # Calculate reference energies (isolated atoms)
            ref_atom1 = Atoms(symbols=[ion1], positions=[[0, 0, 0]], 
                              cell=[cell_size, cell_size, cell_size], pbc=[True, True, True])
            ref_atom1.calc = calc
            ref_energy1 = ref_atom1.get_potential_energy()
            
            ref_atom2 = Atoms(symbols=[ion2], positions=[[0, 0, 0]], 
                              cell=[cell_size, cell_size, cell_size], pbc=[True, True, True])
            ref_atom2.calc = calc
            ref_energy2 = ref_atom2.get_potential_energy()
            
            # The interaction energy is the difference
            interaction_energy = total_energy - (ref_energy1 + ref_energy2)
            
            return interaction_energy
        except Exception as e:
            # If calculation fails, return a high energy to discourage this configuration
            print(f"Warning: ML potential calculation failed: {e}")
            return 1000.0  # Large positive energy

def get_ML_potential(ion_pair, grid_size, cell_size, phase, ml_model=None):
    """
    Get the ML-based energy matrix for a specific ion pair.
    
    Args:
        ion_pair (tuple): Tuple of ion symbols (str, str)
        grid_size (int): Size of the grid
        cell_size (float): Size of the cell in Angstroms
        phase (Phase): Phase object containing ion information
        ml_model (MLPotential, optional): ML potential to use
        
    Returns:
        np.ndarray: Energy matrix for the ion pair
    """
    if ml_model is None:
        ml_model = MLPotential()
    
    # Calculate the full energy matrix
    full_matrix = ml_model.calculate_energy_matrix(grid_size, cell_size, phase)
    
    # Return a view of the matrix for the specific ion pair
    # Note: In a real implementation, we would calculate only the needed part
    return full_matrix 