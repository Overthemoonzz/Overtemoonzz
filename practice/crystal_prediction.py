"""
Generalized Crystal Structure Prediction Tool

This script provides a general framework for predicting crystal structures using
the integer programming approach coupled with energy minimization as described
in the paper by Gusev et al. (Nature, 2023).

The code allows batch prediction of the same species with different parameters
and saves results with sequential naming (BaSnO_1, BaSnO_2, etc.)
"""

from time import time
import os
import shutil
import argparse
import numpy as np

from tabulate import tabulate
import pandas as pd

from ipcsp import root_dir
from ipcsp.integer_program import Allocate
from ipcsp.matrix_generator import Phase
from ase.calculators.gulp import GULP
import ase.io
from copy import deepcopy

# Dictionary of prediction settings for different materials
settings = {
    # Perovskite structure of BaSnO3
    'BaSnO3_1': {'test': False, 'multiple': 1, 'group': 221, 'top': 1, 'grid': 4, 'cell_size': 4.14},
    'BaSnO3_2': {'test': False, 'multiple': 1, 'group': 200, 'top': 1, 'grid': 4, 'cell_size': 4.14},
    'BaSnO3_3': {'test': False, 'multiple': 1, 'group': 195, 'top': 1, 'grid': 4, 'cell_size': 4.14},
    'BaSnO3_4': {'test': False, 'multiple': 2, 'group': 221, 'top': 1, 'grid': 8, 'cell_size': 4.14},
    'BaSnO3_5': {'test': False, 'multiple': 3, 'group': 221, 'top': 1, 'grid': 6, 'cell_size': 4.14},
    
    # Perovskite structure of SrTiO3
    'SrTiO3_1': {'test': True, 'multiple': 1, 'group': 1, 'top': 1, 'grid': 4, 'cell_size': 3.9},
    'SrTiO3_2': {'test': True, 'multiple': 2, 'group': 195, 'top': 1, 'grid': 4, 'cell_size': 3.9},
    'SrTiO3_3': {'test': True, 'multiple': 3, 'group': 221, 'top': 1, 'grid': 8, 'cell_size': 3.9},
    'SrTiO3_4': {'test': True, 'multiple': 3, 'group': 200, 'top': 1, 'grid': 6, 'cell_size': 3.9},
    
    # Bixbyite structure of Y2O3
    'Y2O3_1': {'test': False, 'group': 206, 'top': 1, 'grid': 8, 'cell_size': 10.7, 'multiple': 1},
    'Y2O3_2': {'test': False, 'group': 199, 'top': 1, 'grid': 8, 'cell_size': 10.7, 'multiple': 1},
    'Y2O3_3': {'test': False, 'group': 206, 'top': 1, 'grid': 16, 'cell_size': 10.7, 'multiple': 1},
    
    # Quantum annealing section
    'quantum_BaSnO3': {'test': False, 'group': 221, 'at_dwave': False, 'num_reads': 100,
                     'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                     'annealing_time': 200, 'cell_size': 4.14, 'grid': 4},
    
    'quantum_SrTiO3': {'test': False, 'group': 221, 'at_dwave': False, 'num_reads': 100,
                     'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                     'annealing_time': 200, 'cell_size': 3.9, 'grid': 4},
}

# Material compositions dictionary - mapping of species to their atomic composition
compositions = {
    'BaSnO3': {'Ba': 1, 'Sn': 1, 'O': 3},
    'SrTiO3': {'O': 3, 'Sr': 1, 'Ti': 1},
    'Y2O3': {'Y': 32, 'O': 48},
    # Add more compositions as needed
}

# Phase library mappings
phase_libraries = {
    'BaSnO3': 'BaSnO',
    'SrTiO3': 'SrTiO',
    'Y2O3': 'YSrTiO',  # This uses the potential parameters from YSrTiO
    # Add more mappings as needed
}


def process_results(lib, results, ions_count, test_name, printing=False):
    """
    Process the results from the integer program optimization.
    
    Args:
        lib: Path to the GULP library file
        results: List of ASE Atoms objects representing predicted structures
        ions_count: Dictionary mapping element symbols to their counts
        test_name: Name of the test for saving results
        printing: Whether to print detailed results
        
    Returns:
        float: Energy of the best (lowest energy) structure
    """
    # Create results directory if it doesn't exist
    if not os.path.exists(os.path.join("..", "results")):
        os.mkdir(os.path.join("..", "results"))
    
    # Create test-specific directory
    os.mkdir(os.path.join("..", "results", test_name))
    calc = GULP(keywords='single', library=os.path.join(".", lib))

    # Make a copy of results for storing
    results_ip = deepcopy(results)

    # Compute the number of atoms
    N_atoms = sum(ions_count.values())

    init = [0] * len(results)
    final = [0] * len(results)
    best_val = 0
    best_idx = 0
    
    print("Processing and locally optimising solutions from the integer program\n")
    for idx, cryst in enumerate(results):
        if len(cryst.arrays['positions']) == N_atoms:
            cryst.calc = calc
            init[idx] = cryst.get_potential_energy()
        else:
            print("GULP received a bad solution. Gurobi's implementation of pooling occasionally provides solutions "
                  "that do not satisfy constraints. It should be corrected in future versions of the solver.")

    # Configure GULP for structure optimization
    calc.set(keywords='opti conjugate conp diff comp c6')
    prev_energy = -1000000
    for idx, cryst in enumerate(results):
        if init[idx] < -0.00001:
            if init[idx] - prev_energy > 0.000001:
                prev_energy = init[idx]
                opt = calc.get_optimizer(cryst)
                try:
                    opt.run(fmax=0.05)
                    final[idx] = cryst.get_potential_energy()
                except ValueError:
                    print("One of the relaxations failed using initial energy instead")
                    final[idx] = init[idx]

                if final[idx] < best_val:
                    best_idx = idx
                    best_val = final[idx]

    # Save results with detailed information
    count = 1
    with open(os.path.join("..", "results", test_name, "energies.txt"), "w+") as f:
        for i in range(len(results)):
            if final[i] != 0:
                energy_info = f"Solution{count}: Energy initial: {init[i]} final: {final[i]}"
                print(energy_info)
                f.write(energy_info + "\n")
                
                # Save structures
                ase.io.write(os.path.join("..", "results", test_name, f'solution{count}_lattice.vasp'), results_ip[i])
                ase.io.write(os.path.join("..", "results", test_name, f'solution{count}_minimised.vasp'), results[i])
                count += 1

    # Output the best solution's energy
    cryst = results[best_idx]
    print(f"The lowest found energy is {best_val} eV")
    print(f"The energy per ion is {best_val/N_atoms} eV")
    
    if printing:
        print("Paused, the files can be copied")
        input()

    return best_val


def get_cif_energies(filename, library, format='cif'):
    """
    Calculate the energy of a reference structure from a CIF file.
    
    Args:
        filename: Name of the CIF file
        library: Path to the GULP library file
        format: File format (default: 'cif')
        
    Returns:
        float: Energy of the reference structure
    """
    filedir = root_dir / 'structures/'
    try:
        cryst = ase.io.read(os.path.join(".", filedir / filename), format=format, parallel=False)
        calc = GULP(keywords='conp', library=library)
        calc.set(keywords='opti conjugate conp diff comp c6')
        opt = calc.get_optimizer(cryst)
        opt.run(fmax=0.05)
        energy = cryst.get_potential_energy()
        
        print(f"The energy of {filename} is equal to {energy} eV")
        return energy
    except Exception as e:
        print(f"Error calculating energy for {filename}: {e}")
        return None


def benchmark():
    """
    Run benchmark tests for crystal structure prediction on multiple materials
    with different parameters.
    """
    # Preparing results folder
    shutil.rmtree(os.path.join("..", "results"), ignore_errors=True)
    os.mkdir(os.path.join("..", "results"))

    # For focused testing of a single material configuration:
    '''
    # Single test selector example
    for key in settings.keys():
        settings[key]['test'] = False
    
    settings['BaSnO3_2']['test'] = True
    '''
    
    # DataFrame to store summary results
    df_summary = pd.DataFrame(columns=['name', 'grid', 'group', 'best_E', 'expected_E', 'time'])

    # Process BaSnO3 predictions with different parameters
    for i in range(1, 6):  # Process tests 1 through 5
        if settings[f'BaSnO3_{i}']['test']:
            print(f"\n\n\n========== Predicting BaSnO3 (perovskite structure) - Test {i} ==========")
            print(settings[f'BaSnO3_{i}'])

            BaSnO = Phase('BaSnO')  # Initialize the phase with potential parameters

            multiple = settings[f'BaSnO3_{i}']['multiple']
            cell_size = settings[f'BaSnO3_{i}']['cell_size'] * multiple
            
            # Scale the composition based on the multiple parameter
            base_ions_count = compositions['BaSnO3']
            ions_count = {ion: count * (multiple ** 3) for ion, count in base_ions_count.items()}
            
            print(f"Running with cell size: {cell_size} Å")
            print(f"Composition: {ions_count}")

            start = time()
            
            # Create allocation object for integer programming
            allocation = Allocate(ions_count, grid_size=settings[f'BaSnO3_{i}']['grid'], 
                                 cell_size=cell_size, phase=BaSnO)

            # Run optimization with space group symmetry
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(
                group=settings[f'BaSnO3_{i}']['group'],
                PoolSolutions=settings[f'BaSnO3_{i}']['top'],
                TimeLimit=0
            )

            # Process and analyze results
            best_energy = process_results(
                lib=BaSnO.filedir / 'BaSnO/buck.lib', 
                results=results,
                ions_count=ions_count, 
                test_name=f'BaSnO_{i}'
            )

            # Calculate reference energy from CIF file
            energy = get_cif_energies(
                filename='BaSnO3.cif', 
                library=BaSnO.filedir / 'BaSnO/buck.lib'
            )
            
            # Scale reference energy for supercells
            if multiple > 1:
                energy = energy * multiple ** 3
                print(f"For the given multiple it is equal to {energy} eV")

            # Calculate total runtime
            end = time()
            print(f"It took {end - start} seconds including IP and data generation")

            # Add results to summary dataframe
            df_summary = df_summary.append({
                'name': f'BaSnO3_{i}', 
                'grid': settings[f'BaSnO3_{i}']['grid'],
                'group': settings[f'BaSnO3_{i}']['group'], 
                'best_E': best_energy,
                'expected_E': energy, 
                'time': runtime
            }, ignore_index=True)
    
    # Process SrTiO3 predictions with different parameters
    for i in range(1, 5):  # Process tests 1 through 4
        if settings[f'SrTiO3_{i}']['test']:
            print(f"\n\n\n========== Predicting SrTiO3 (perovskite structure) - Test {i} ==========")
            print(settings[f'SrTiO3_{i}'])

            SrTiO = Phase('SrTiO')  # Initialize the phase with potential parameters

            multiple = settings[f'SrTiO3_{i}']['multiple']
            cell_size = settings[f'SrTiO3_{i}']['cell_size'] * multiple
            
            # Scale the composition based on the multiple parameter
            base_ions_count = compositions['SrTiO3']
            ions_count = {ion: count * (multiple ** 3) for ion, count in base_ions_count.items()}
            
            print(f"Running with cell size: {cell_size} Å")
            print(f"Composition: {ions_count}")

            start = time()
            
            # Create allocation object for integer programming
            allocation = Allocate(ions_count, grid_size=settings[f'SrTiO3_{i}']['grid'], 
                                 cell_size=cell_size, phase=SrTiO)

            # Run optimization with space group symmetry
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(
                group=settings[f'SrTiO3_{i}']['group'],
                PoolSolutions=settings[f'SrTiO3_{i}']['top'],
                TimeLimit=0
            )

            # Process and analyze results
            best_energy = process_results(
                lib=SrTiO.filedir / 'SrTiO/buck.lib', 
                results=results,
                ions_count=ions_count, 
                test_name=f'SrTiO_{i}'
            )

            # Calculate reference energy from CIF file
            energy = get_cif_energies(
                filename='SrTiO3.cif', 
                library=SrTiO.filedir / 'SrTiO/buck.lib'
            )
            
            # Scale reference energy for supercells
            if multiple > 1:
                energy = energy * multiple ** 3
                print(f"For the given multiple it is equal to {energy} eV")

            # Calculate total runtime
            end = time()
            print(f"It took {end - start} seconds including IP and data generation")

            # Add results to summary dataframe
            df_summary = df_summary.append({
                'name': f'SrTiO3_{i}', 
                'grid': settings[f'SrTiO3_{i}']['grid'],
                'group': settings[f'SrTiO3_{i}']['group'], 
                'best_E': best_energy,
                'expected_E': energy, 
                'time': runtime
            }, ignore_index=True)
    
    # Process Y2O3 predictions with different parameters
    for i in range(1, 4):  # Process tests 1 through 3
        if settings[f'Y2O3_{i}']['test']:
            print(f"\n\n\n========== Predicting Y2O3 (bixbyite structure) - Test {i} ==========")
            print(settings[f'Y2O3_{i}'])

            YSrTiO = Phase('YSrTiO')  # Initialize the phase with potential parameters

            multiple = settings[f'Y2O3_{i}'].get('multiple', 1)
            cell_size = settings[f'Y2O3_{i}']['cell_size'] * multiple
            
            # Use predefined composition for Y2O3
            # Y2O3 composition is already defined for the full unit cell
            ions_count = compositions['Y2O3']
            
            print(f"Running with cell size: {cell_size} Å")
            print(f"Composition: {ions_count}")

            start = time()
            
            # Create allocation object for integer programming
            allocation = Allocate(ions_count, grid_size=settings[f'Y2O3_{i}']['grid'], 
                                 cell_size=cell_size, phase=YSrTiO)

            # Run optimization with space group symmetry
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(
                group=settings[f'Y2O3_{i}']['group'],
                PoolSolutions=settings[f'Y2O3_{i}']['top'],
                TimeLimit=0
            )

            # Process and analyze results
            best_energy = process_results(
                lib=YSrTiO.filedir / 'YSrTiO/buck.lib', 
                results=results,
                ions_count=ions_count, 
                test_name=f'Y2O3_{i}'
            )

            # Calculate reference energy from CIF file
            energy = get_cif_energies(
                filename='Y2O3.cif', 
                library=YSrTiO.filedir / 'YSrTiO/buck.lib'
            )

            # Calculate total runtime
            end = time()
            print(f"It took {end - start} seconds including IP and data generation")

            # Add results to summary dataframe
            df_summary = df_summary.append({
                'name': f'Y2O3_{i}', 
                'grid': settings[f'Y2O3_{i}']['grid'],
                'group': settings[f'Y2O3_{i}']['group'], 
                'best_E': best_energy,
                'expected_E': energy, 
                'time': runtime
            }, ignore_index=True)
    
    # Process quantum annealing tests
    quantum_df = pd.DataFrame(columns=['name', 'dwave', 'best_E', 'expected_E'])
    
    # Quantum test for BaSnO3
    if settings['quantum_BaSnO3']['test']:
        print("\n\n\n========== Predicting BaSnO3 (perovskite) using quantum annealer ==========")
        BaSnO = Phase('BaSnO')
        multiple = settings['quantum_BaSnO3']['multiple']
        cell_size = settings['quantum_BaSnO3'].get('cell_size', 4.14) * multiple
        
        # Scale the composition
        base_ions_count = compositions['BaSnO3']
        ions_count = {ion: count * (multiple ** 3) for ion, count in base_ions_count.items()}

        start = time()
        allocation = Allocate(ions_count, grid_size=settings['quantum_BaSnO3']['grid'], 
                            cell_size=cell_size, phase=BaSnO)

        best_energy, target_energy = allocation.optimize_qubo(
            group=settings['quantum_BaSnO3']['group'],
            at_dwave=settings['quantum_BaSnO3']['at_dwave'],
            num_reads=settings['quantum_BaSnO3']['num_reads'],
            infinity_placement=settings['quantum_BaSnO3']['infinity_placement'],
            infinity_orbit=settings['quantum_BaSnO3']['infinity_orbit'],
            annealing_time=settings['quantum_BaSnO3']['annealing_time']
        )

        energy = get_cif_energies(
            filename='BaSnO3.cif', 
            library=BaSnO.filedir / 'BaSnO/buck.lib'
        )

        # Add results to quantum dataframe
        quantum_df = quantum_df.append({
            'name': 'quantum_BaSnO3',
            'dwave': settings['quantum_BaSnO3']['at_dwave'],
            'best_E': best_energy, 
            'expected_E': target_energy
        }, ignore_index=True)

        end = time()
        print(f"It took {end - start} seconds")
    
    # Quantum test for SrTiO3
    if settings['quantum_SrTiO3']['test']:
        print("\n\n\n========== Predicting SrTiO3 (perovskite) using quantum annealer ==========")
        SrTiO = Phase('SrTiO')
        multiple = settings['quantum_SrTiO3']['multiple']
        cell_size = settings['quantum_SrTiO3'].get('cell_size', 3.9) * multiple
        
        # Scale the composition
        base_ions_count = compositions['SrTiO3']
        ions_count = {ion: count * (multiple ** 3) for ion, count in base_ions_count.items()}

        start = time()
        allocation = Allocate(ions_count, grid_size=settings['quantum_SrTiO3']['grid'], 
                            cell_size=cell_size, phase=SrTiO)

        best_energy, target_energy = allocation.optimize_qubo(
            group=settings['quantum_SrTiO3']['group'],
            at_dwave=settings['quantum_SrTiO3']['at_dwave'],
            num_reads=settings['quantum_SrTiO3']['num_reads'],
            infinity_placement=settings['quantum_SrTiO3']['infinity_placement'],
            infinity_orbit=settings['quantum_SrTiO3']['infinity_orbit'],
            annealing_time=settings['quantum_SrTiO3']['annealing_time']
        )

        energy = get_cif_energies(
            filename='SrTiO3.cif', 
            library=SrTiO.filedir / 'SrTiO/buck.lib'
        )

        # Add results to quantum dataframe
        quantum_df = quantum_df.append({
            'name': 'quantum_SrTiO3',
            'dwave': settings['quantum_SrTiO3']['at_dwave'],
            'best_E': best_energy, 
            'expected_E': target_energy
        }, ignore_index=True)

        end = time()
        print(f"It took {end - start} seconds")
    
    # Save summary table
    with open(os.path.join("..", "results", "summary.txt"), "w+") as f:
        print("Non-heuristic optimisation using Gurobi with subsequent local minimisation:", file=f)
        print(tabulate(df_summary, headers=["Test name", "Discretisation g", "Space group",
                                            "Best energy (eV)", "Target energy (eV)", "IP solution time (sec)"],
                      tablefmt='github', showindex=False), file=f)
        
        # Add quantum results if available
        if not quantum_df.empty:
            print("\n\n\n\n\n Quantum annealing for the periodic lattice atom allocation.\n", file=f)
            print(tabulate(quantum_df, headers=["Test name", "D-Wave", "Best energy (eV)", "Target energy (eV)"],
                        tablefmt='github', showindex=False), file=f)


def parse_arguments():
    """Parse command line arguments for customizing the batch prediction."""
    parser = argparse.ArgumentParser(description='Crystal Structure Prediction with Batch Testing')
    parser.add_argument('--enable-all', action='store_true', help='Enable all tests')
    parser.add_argument('--enable', type=str, nargs='+', help='Enable specific tests by name (e.g., BaSnO3_1)')
    parser.add_argument('--disable', type=str, nargs='+', help='Disable specific tests by name')
    parser.add_argument('--quantum', action='store_true', help='Enable quantum annealing tests')
    parser.add_argument('--material', type=str, help='Run all tests for a specific material (e.g., BaSnO3, SrTiO3, Y2O3)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure which tests to run based on command line arguments
    if args.enable_all:
        for key in settings:
            settings[key]['test'] = True
    
    if args.material:
        for key in settings:
            if key.startswith(args.material):
                settings[key]['test'] = True
    
    if args.enable:
        for test_name in args.enable:
            if test_name in settings:
                settings[test_name]['test'] = True
    
    if args.disable:
        for test_name in args.disable:
            if test_name in settings:
                settings[test_name]['test'] = False
    
    if args.quantum:
        for key in settings:
            if key.startswith('quantum_'):
                settings[key]['test'] = True
    
    # Run the benchmark
    benchmark()