"""
Generalized Crystal Structure Prediction Tool

This script provides a general framework for predicting crystal structures using
the integer programming approach coupled with energy minimization as described
in the paper by Gusev et al. (Nature, 2023) and based on methods by Woodley et al. (1999).

Usage:
    python crystal_predictor.py --species BaSnO3 --space-group 221 --cell-size 4.14346542 --grid-size 6

Or by modifying the parameters directly in this script.
"""

import os
import glob
import numpy as np
import argparse
from ase.io import read, write
import ase.io
import time

from ipcsp.integer_program import Allocate
from ipcsp.matrix_generator import Phase
from ipcsp import root_dir
from ase.calculators.gulp import GULP

# Default configuration parameters - change these or use command line arguments
DEFAULT_CONFIG = {
    "species": "BaSnO3",        # Target species
    "space_group": "221",       # Space group number
    "grid_size": 8,             # Grid size for discretization
    "multiple": 3,              # the number of repeats of the unit cell per direction, essentially, we are predicting multiple
    "cell_size": 71.13628093,    # Cell parameter in Angstroms
    "time_limit": 600,          # Time limit for optimization in seconds
    "pool_solutions": 1,        # Number of solutions to generate
    "composition": {            # Chemical composition (modify based on your species)
        "BaSnO3": {'Ba': 1, 'Sn': 1, 'O': 3},
        "SrTiO3": {'Sr': 1, 'Ti': 1, 'O': 3},
        "MgO": {'Mg': 4, 'O': 4},
        "Y2O3": {'Y': 32, 'O': 48},
        # Add more compositions as needed
    },
    "phase_library": {          # Mapping of species to phase library names
        "BaSnO3": "BaSnO",
        "SrTiO3": "SrTiO",
        "MgO": "MgO",
        "Y2O3": "YSrTiO",      # This uses the potential parameters from YSrTiO
        # Add more mappings as needed
    },
    "space_group_name": {       # Human-readable space group names
        "221": "Pm-3m",
        "225": "Fm-3m",
        "206": "Ia3",
        "200": "Pm-3",
        # Add more space groups as needed
    },
    "expected_bond_distances": {  # Expected bond distances for verification
        "BaSnO3": {"cation1-O": 2.93034, "cation2-O": 2.07173271},  # Ba-O, Sn-O
        "SrTiO3": {"cation1-O": 2.7572, "cation2-O": 1.9495},  # Sr-O, Ti-O
        "MgO": {"cation1-O": 2.09700140},
        # Add more expected distances as needed
    },
    "cation_names": {           # Names of cations for distance analysis
        "BaSnO3": ["Ba", "Sn"],
        "SrTiO3": ["Sr", "Ti"],
        "MgO": ["Mg"],
        # Add more cation lists as needed
    }
}

def get_cif_energies(filename, library, format='cif'):
    """
    Calculate the energy of a structure from a CIF file using GULP.
    
    Args:
        filename: Path to the CIF file
        library: Path to the potential library file
        format: File format (default: 'cif')
        
    Returns:
        Energy of the optimized structure in eV
    """
    filedir = root_dir / 'structures/'
    try:
        cryst = ase.io.read(os.path.join(".", filedir / filename), format=format, parallel=False)
        calc = GULP(keywords='conp', library=library)
        calc.set(keywords='opti conjugate conp diff comp c6')
        opt = calc.get_optimizer(cryst)
        opt.run(fmax=0.05)
        energy = cryst.get_potential_energy()
        
        print(f"The energy of {filename} is equal to {energy:.6f} eV")
        return energy
    except Exception as e:
        print(f"Error calculating energy for {filename}: {e}")
        return None

def prepare_directories(species):
    """
    Create necessary directories for results.
    
    Args:
        species: Target species name
        
    Returns:
        Dictionary with paths to the created directories
    """
    # Create main results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Create species-specific results directory
    species_dir = os.path.join(results_dir, species)
    if not os.path.exists(species_dir):
        os.makedirs(species_dir)
        print(f"Created species directory: {species_dir}")
    
    # Check for structures directory
    structures_dir = "structures"
    if not os.path.exists(structures_dir):
        print(f"Warning: Structures directory '{structures_dir}' does not exist!")
    
    return {
        "results": results_dir,
        "species": species_dir,
        "structures": structures_dir
    }

def analyze_structure(structure, species, cell_size, config):
    """
    Analyze the predicted structure, calculating bond distances.
    
    Args:
        structure: ASE Atoms object of the predicted structure
        species: Target species name
        cell_size: Cell parameter in Angstroms
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {}
    
    # Get cation names for this species
    cation_names = config["cation_names"].get(species, [])
    
    # Calculate distances between cations and O
    for idx, cation in enumerate(cation_names):
        distances = []
        for i, atom_i in enumerate(structure):
            if atom_i.symbol == cation:
                for j, atom_j in enumerate(structure):
                    if atom_j.symbol == 'O':
                        dist = structure.get_distance(i, j, mic=True)
                        # Consider only reasonable distances (adjust cutoff if needed)
                        if dist < 3.5:
                            distances.append(dist)
        
        if distances:
            avg_dist = np.mean(distances)
            analysis[f"{cation}-O_avg"] = avg_dist
            print(f"Average {cation}-O distance: {avg_dist:.3f} Å")
    
    # Calculate expected values if available
    if species in config["expected_bond_distances"]:
        for idx, cation in enumerate(cation_names):
            key = f"cation{idx+1}-O"
            if key in config["expected_bond_distances"][species]:
                expected = config["expected_bond_distances"][species][key]
                analysis[f"{cation}-O_expected"] = expected
                print(f"Expected {cation}-O distance: {expected:.3f} Å")
    
    return analysis

def predict_crystal_structure(config):
    """
    Main function to predict crystal structure using the IPCSP method.
    
    Args:
        config: Configuration dictionary with parameters
        
    Returns:
        Tuple of predicted structure and reference structure
    """
    species = config["species"]
    space_group = config["space_group"]
    grid_size = config["grid_size"]
    base_cell_size = config["cell_size"]
    multiple = config["multiple"]
    time_limit = config["time_limit"]
    pool_solutions = config["pool_solutions"]
    
    # Scale cell size and adjust composition based on the multiple parameter
    cell_size = base_cell_size * multiple
    
    # Get composition for this species
    if species not in config["composition"]:
        raise ValueError(f"Composition for {species} not defined in configuration")
    
    # Scale the composition by the multiple^3 factor
    base_ions_count = config["composition"][species]
    ions_count = {ion: count * (multiple ** 3) for ion, count in base_ions_count.items()}
    
    # Get phase library name for this species
    if species not in config["phase_library"]:
        raise ValueError(f"Phase library for {species} not defined in configuration")
    phase_name = config["phase_library"][species]
    
    # Get space group name if available
    space_group_name = config["space_group_name"].get(space_group, "")
    
    print(f"====== Testing IPCSP on {species} (space group {space_group} {space_group_name}) ======")
    print(f"Grid size: {grid_size}, Base cell size: {base_cell_size} Å")
    print(f"Supercell multiplier: {multiple}, Scaled cell size: {cell_size} Å")
    print(f"Base composition: {base_ions_count}")
    print(f"Scaled composition: {ions_count}")
    print(f"Space group: {space_group} {space_group_name}")
    
    # Create directories
    dirs = prepare_directories(species)
    
    # Start timing
    start_time = time.time()
    
    # Initialize the phase
    try:
        phase = Phase(phase_name)
    except Exception as e:
        print(f"Error initializing phase {phase_name}: {e}")
        print("Make sure the necessary potential files exist in the data directory.")
        return None, None
    
    try:
        # Create allocation object
        allocation = Allocate(ions_count, grid_size=grid_size, cell_size=cell_size, phase=phase)
        
        # Run optimization
        print(f"Running optimization with space group {space_group}...")
        results, runtime, energy = allocation.optimize_cube_symmetry_ase(
            group=space_group,
            PoolSolutions=pool_solutions,
            TimeLimit=time_limit
        )
        
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {runtime:.2f} seconds (total time: {total_time:.2f} seconds)")
        print(f"Energy: {energy:.6f} eV")
        
        # Process results
        if isinstance(results, list) and len(results) > 0:
            predicted_structure = results[0]
            
            # Save predicted structure
            predicted_path = os.path.join(dirs["species"], f"{species.lower()}_predicted.cif")
            write(predicted_path, predicted_structure)
            print(f"\nPredicted structure saved as '{predicted_path}'")
            
            # Set up GULP calculator for energy calculation
            calc = GULP(keywords='single', library=phase.filedir / f'{phase_name}/buck.lib')
            predicted_structure.calc = calc
            
            try:
                initial_energy = predicted_structure.get_potential_energy()
                
                # Calculate energy per atom for easier comparison
                n_atoms = len(predicted_structure)
                initial_energy_per_atom = initial_energy / n_atoms
                print(f"Initial energy per atom: {initial_energy_per_atom:.6f} eV/atom")
                
                # Optimize structure with GULP
                calc.set(keywords='opti conjugate conp diff comp c6')
                opt = calc.get_optimizer(predicted_structure)
                opt.run(fmax=0.05)
                final_energy = predicted_structure.get_potential_energy()
                final_energy_per_atom = final_energy / n_atoms
                print(f"\nInitial energy: {initial_energy:.6f} eV ({initial_energy_per_atom:.6f} eV/atom)")
                print(f"Final energy after optimization: {final_energy:.6f} eV ({final_energy_per_atom:.6f} eV/atom)")
            except ValueError as e:
                print(f"Relaxation failed: {e}")
                print("Using initial energy instead")
                final_energy = initial_energy
                final_energy_per_atom = initial_energy_per_atom
            
            # Save optimized structure
            optimized_path = os.path.join(dirs["species"], f"{species.lower()}_optimized.cif")
            write(optimized_path, predicted_structure)
            print(f"Optimized structure saved as '{optimized_path}'")
            
            # Compare with reference structure if available
            reference_structure = None
            reference_filename = f"{species}.cif"
            reference_path = os.path.join(dirs["structures"], reference_filename)
            
            try:
                # First load the reference structure
                reference_structure = ase.io.read(reference_path)
                print(f"Reference structure loaded from {reference_path}")
                
                # Then calculate its energy
                reference_energy = get_cif_energies(
                    filename=reference_filename, 
                    library=phase.filedir / f'{phase_name}/buck.lib'
                )
                if multiple > 1:
                    reference_energy = reference_energy * multiple ** 3
                    print("For the given multiple it is equal to ", reference_energy, "eV")
                if reference_energy is not None:
                    reference_energy_per_atom = reference_energy / n_atoms
                    print(f"Reference structure energy: {reference_energy:.6f} eV ({reference_energy_per_atom:.6f} eV/atom)")
                    
                    # Calculate energy difference
                    energy_diff = final_energy - reference_energy
                    energy_diff_per_atom = final_energy_per_atom - reference_energy_per_atom
                    print(f"Energy difference (predicted - reference): {energy_diff:.6f} eV ({energy_diff_per_atom:.6f} eV/atom)")
                
            except Exception as e:
                print(f"Could not process reference structure: {e}")
                reference_structure = None
                reference_energy = None
                reference_energy_per_atom = None
                energy_diff = None
                energy_diff_per_atom = None
            
            # Save energy results
            energy_path = os.path.join(dirs["species"], "energies.txt")
            with open(energy_path, 'w') as f:
                f.write(f"Species: {species}\n")
                f.write(f"Space group: {space_group} {space_group_name}\n")
                f.write(f"Cell size: {cell_size} Å\n\n")
                f.write(f"Solution1:  Energy initial:  {initial_energy:.8f}  final:  {final_energy:.8f}\n")
                f.write(f"Per atom:   Energy initial:  {initial_energy_per_atom:.8f}  final:  {final_energy_per_atom:.8f}\n")
                if reference_energy is not None:
                    f.write(f"Reference energy: {reference_energy:.8f} ({reference_energy_per_atom:.8f} eV/atom)\n")
                    f.write(f"Energy difference: {energy_diff:.8f} ({energy_diff_per_atom:.8f} eV/atom)\n")
            print(f"Energy results saved to {energy_path}")
            
            # Analyze the structure
            analysis = analyze_structure(predicted_structure, species, cell_size, config)
            
            # Save a full analysis report
            report_path = os.path.join(dirs["species"], f"{species.lower()}_analysis.txt")
            with open(report_path, 'w') as f:
                f.write(f"{species} Structure Prediction Analysis\n")
                f.write("="*40 + "\n\n")
                f.write(f"Space group: {space_group} {space_group_name}\n")
                f.write(f"Cell parameter: {cell_size:.6f} Å\n")
                f.write(f"Optimization completed in {runtime:.2f} seconds\n")
                f.write(f"Initial IP energy: {energy:.6f} eV\n\n")
                f.write(f"Initial energy: {initial_energy:.6f} eV ({initial_energy_per_atom:.6f} eV/atom)\n")
                f.write(f"Final energy after optimization: {final_energy:.6f} eV ({final_energy_per_atom:.6f} eV/atom)\n")
                if reference_energy is not None:
                    f.write(f"Reference structure energy: {reference_energy:.6f} eV ({reference_energy_per_atom:.6f} eV/atom)\n")
                    f.write(f"Energy difference: {energy_diff:.6f} eV ({energy_diff_per_atom:.6f} eV/atom)\n\n")
                
                f.write("Bond distances:\n")
                for key, value in analysis.items():
                    if key.endswith("_avg"):
                        cation = key.split("-")[0]
                        f.write(f"Average {cation}-O distance: {value:.3f} Å\n")
                        if f"{cation}-O_expected" in analysis:
                            expected = analysis[f"{cation}-O_expected"]
                            f.write(f"Expected {cation}-O distance: {expected:.3f} Å\n")
                            f.write(f"Difference: {abs(value - expected):.3f} Å\n")
                
                f.write("\nStructure verification:\n")
                f.write(f"1. Structure should have the expected arrangement for space group {space_group} {space_group_name}\n")
                f.write("2. Bond distances should be close to expected values\n")
                f.write("3. The energy should be negative and close to the reference\n")
            
            print(f"Analysis report saved to {report_path}")
            
            return predicted_structure, reference_structure
        else:
            print("Optimization failed to return a valid structure")
            
            # Save error report
            error_path = os.path.join(dirs["species"], "error_report.txt")
            with open(error_path, 'w') as f:
                f.write(f"{species} Structure Prediction Failed\n")
                f.write("="*40 + "\n\n")
                f.write(f"Optimization runtime: {runtime:.2f} seconds\n")
                f.write(f"Energy returned: {energy}\n")
                f.write("No valid structure was produced.\n")
            print(f"Error report saved to {error_path}")
            
            return None, None
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crystal Structure Prediction')
    parser.add_argument('--species', type=str, help='Target chemical species')
    parser.add_argument('--space-group', type=str, help='Space group number')
    parser.add_argument('--grid-size', type=int, help='Grid size for discretization')
    parser.add_argument('--cell-size', type=float, help='Unit cell parameter in Angstroms')
    parser.add_argument('--multiple', type=int, help='Supercell multiplier (1=unit cell, 2=2×2×2 supercell, etc.)')
    parser.add_argument('--time-limit', type=int, help='Time limit for optimization in seconds')
    parser.add_argument('--pool-solutions', type=int, help='Number of solutions to generate')
    
    return parser.parse_args()

def main():
    """Main function to run crystal structure prediction."""
    # Get command line arguments
    args = parse_arguments()
    
    # Create configuration by starting with defaults
    config = DEFAULT_CONFIG.copy()
    
    # Update config with command line arguments if provided
    if args.species:
        config["species"] = args.species
    if args.space_group:
        config["space_group"] = args.space_group
    if args.grid_size:
        config["grid_size"] = args.grid_size
    if args.cell_size:
        config["cell_size"] = args.cell_size
    if args.multiple:
        config["multiple"] = args.multiple
    if args.time_limit:
        config["time_limit"] = args.time_limit
    if args.pool_solutions:
        config["pool_solutions"] = args.pool_solutions
    
    # Run the prediction
    predicted, reference = predict_crystal_structure(config)
    
    if predicted is not None:
        print("\nVerification:")
        print(f"1. Structure should have the expected arrangement for space group {config['space_group']}")
        print("2. Bond distances should be close to expected values")
        print("3. The energy should be negative and close to the reference")
    
    return predicted, reference

if __name__ == "__main__":
    predicted, reference = main()
