"""
This is the code to reproduce Table 1 and assess performance of a D-Wave quantum annealer for CSP.
The latter by default uses simulated annealing implementation on a classical computer provided by
D-Wave and a quantum annealer can be accessed after a registration with a single parameter change.

Originally, a GULP ASE calculator was used to load a force field.
This version replaces it with M3GNET for energy evaluation and structure relaxation.

M3GNET example usage:
--------------------------------------------------
from __future__ import annotations
import warnings
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
from matgl.ext.ase import Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
relaxer = Relaxer(potential=pot)
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
relax_results = relaxer.relax(struct, fmax=0.01)
final_structure = relax_results["final_structure"]
final_energy = relax_results["trajectory"].energies[-1]
print(final_structure)
print(f"The final energy is {float(final_energy):.3f} eV.")
--------------------------------------------------

Below is the complete modified script.
"""

from time import time
import os
import shutil
from tabulate import tabulate
import pandas as pd
from pathlib import Path

from ipcsp import root_dir
from ipcsp.integer_program import Allocate
from ipcsp.matrix_generator import Phase
import ase.io
from copy import deepcopy

# Import M3GNET-related modules and set up warnings
import warnings
warnings.simplefilter("ignore")
import matgl
from matgl.ext.ase import Relaxer
from pymatgen.io.ase import AseAtomsAdaptor

# Settings dictionary remains unchanged (except that the 'lib' parameters are now ignored)
settings = {
    # Integer programming section tests
    'SrTiO_1': {'test': True, 'multiple': 1, 'group': 1, 'top': 1, 'grid': 4, 'use_ml': False},
    'SrTiO_2': {'test': False, 'multiple': 2, 'group': 195, 'top': 1, 'grid': 8, 'use_ml': False},
    'SrTiO_3': {'test': False, 'multiple': 3, 'group': 221, 'top': 1, 'grid': 6, 'use_ml': False},
    'SrTiO_4': {'test': False, 'multiple': 3, 'group': 200, 'top': 1, 'grid': 6, 'use_ml': False},
    'SrTiO_5': {'test': False, 'multiple': 3, 'group': 195, 'top': 1, 'grid': 6, 'use_ml': False},
    'Y2O3_1': {'test': False, 'group': 206, 'top': 1, 'grid': 8},
    'Y2O3_2': {'test': False, 'group': 199, 'top': 1, 'grid': 8},
    'Y2O3_3': {'test': False, 'group': 206, 'top': 1, 'grid': 16},
    'Y2Ti2O7_1': {'test': False, 'group': 227, 'top': 2, 'grid': 8},
    'Y2Ti2O7_2': {'test': False, 'group': 227, 'top': 1, 'grid': 16},
    'MgAl2O4_1': {'test': False, 'group': 227, 'top': 1, 'grid': 8},
    'MgAl2O4_2': {'test': False, 'group': 227, 'top': 1, 'grid': 16},
    'MgAl2O4_3': {'test': False, 'group': 196, 'top': 1, 'grid': 8},
    'MgAl2O4_4': {'test': False, 'group': 195, 'top': 20, 'grid': 8},
    'Ca3Al2Si3O12_1': {'test': False, 'group': 230, 'top': 1, 'grid': 16},
    'Ca3Al2Si3O12_2': {'test': False, 'group': 206, 'top': 1, 'grid': 8},
    # Quantum annealer section
    'quantum_SrO': {'test': False, 'group': 195, 'at_dwave': False, 'num_reads': 100,
                    'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                    'annealing_time': 200},
    'quantum_SrTiO': {'test': False, 'group': 221, 'at_dwave': False, 'num_reads': 100,
                       'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                       'annealing_time': 200},
    'quantum_ZnS': {'test': False, 'group': 195, 'at_dwave': False, 'num_reads': 200,
                    'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                    'annealing_time': 200},
    'quantum_ZrO2': {'test': False, 'group': 198, 'at_dwave': False, 'num_reads': 300,
                     'multiple': 1, 'infinity_placement': 50, 'infinity_orbit': 50,
                     'annealing_time': 1000},
    # Add an ML-enabled test
    'SrTiO_1_ml': {'test': False, 'multiple': 1, 'group': 1, 'top': 1, 'grid': 4, 'use_ml': True},
    'SrTiO_2_ml': {'test': False, 'multiple': 2, 'group': 195, 'top': 1, 'grid': 8, 'use_ml': True},
}

# file path
results_dir = Path('/home/yanghn/Application/src/work')
output_dir = Path('/home/yanghn/Application/src/work')

def process_results(lib, results, ions_count, test_name, printing=False):
    """
    Process and relax solutions from the integer program using M3GNET.
    The 'lib' parameter is no longer used.
    """
    # Create test-specific directory
    test_dir = results_dir / "results" / test_name
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize the M3GNET relaxer (load the PES model)
    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    relaxer = Relaxer(potential=pot)
    
    # Stash the original IP allocations for future reference
    results_ip = deepcopy(results)
    
    # Compute the total number of atoms
    N_atoms = sum(ions_count.values())
    
    init = [None] * len(results)
    final = [None] * len(results)
    relax_results_list = [None] * len(results)
    best_val = float('inf')
    best_idx = 0
    
    print("Processing and relaxing solutions using M3GNET\n")
    for idx, cryst in enumerate(results):
        if len(cryst.arrays['positions']) == N_atoms:
            # Convert the ASE Atoms to a pymatgen Structure
            struct = AseAtomsAdaptor.get_structure(cryst)
            relax_results = relaxer.relax(struct, fmax=0.05)
            relax_results_list[idx] = relax_results
            init_energy = relax_results["trajectory"].energies[0]
            final_energy = relax_results["trajectory"].energies[-1]
            init[idx] = init_energy
            final[idx] = final_energy
            
            if final_energy < best_val:
                best_val = final_energy
                best_idx = idx
        else:
            print("M3GNET received a bad solution. The structure may not satisfy constraints.")
    
    count = 1
    with open(test_dir / "energies.txt", "w+") as f:
        for i in range(len(results)):
            if final[i] is not None:
                print(f"Solution{count}: Energy initial: {init[i]}  final: {final[i]}")
                print(f"Solution{count}: Energy initial: {init[i]}  final: {final[i]}", file=f)
                # Save the original lattice (IP allocation)
                ase.io.write(os.path.join(results_dir, "results", test_name, f'solution{count}_predicted.vasp'), results_ip[i])
                # Convert the relaxed final structure (pymatgen Structure) back to ASE Atoms and save it
                final_structure = relax_results_list[i]["final_structure"]
                final_atoms = AseAtomsAdaptor.get_atoms(final_structure)
                ase.io.write(os.path.join(results_dir, "results", test_name, f'solution{count}_minimised.vasp'), final_atoms)
                count += 1
    
    print("The lowest found energy is", best_val, "eV")
    
    if printing:
        print("Paused, the files can be copied")
        input()
    
    return best_val


def get_cif_energies(filename, format='cif'):
    """
    Get the energy of a structure loaded from a CIF file using M3GNET.
    """
    filedir = root_dir / 'structures/'
    
    # Use os.path.join for reliable path construction
    file_path = os.path.join(filedir, filename)
    print(f"Loading structure from: {file_path}")
    
    cryst = ase.io.read(file_path, format=format, parallel=False)
    struct = AseAtomsAdaptor.get_structure(cryst)
    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    relaxer = Relaxer(potential=pot)
    relax_results = relaxer.relax(struct, fmax=0.05)
    energy = relax_results["trajectory"].energies[-1]
    print("The energy of", filename, "is equal to", energy, "eV")

    return energy



def benchmark():
    if not os.path.exists(output_dir / 'results'):
        os.makedirs(output_dir / 'results')
    else:
        shutil.rmtree(output_dir / 'results')
        os.makedirs(output_dir / 'results')

    # Test for different libraries
    libraries = ['SrTiO']  # , 'Y2O3', 'MgAl2O4', 'Ca3Al2Si3O12']

    # Initialize the results data frames
    df_summary = pd.DataFrame(columns=['name', 'grid', 'group', 'best_E', 'expected_E', 'time'])
    
    # Add argument parser for ML potential
    import argparse
    parser = argparse.ArgumentParser(description='Run IPCSP benchmarks')
    parser.add_argument('--use-ml', action='store_true', help='Use ML potentials for short-range interactions')
    parser.add_argument('--ml-model', type=str, default="M3GNet-MP-2021.2.8-PES", 
                        help='MatGL model name for ML potentials')
    args = parser.parse_args()
    
    if args.use_ml:
        print(f"ML potentials enabled with model: {args.ml_model}")

    # Integer programming tests
    for lib in libraries:
        tests = []
        for test in settings.keys():
            if test.startswith(lib) and settings[test]['test']:
                tests.append(test)
        print('Tests: ', tests)
        print()

        grid_calculated = False
        for test_name in tests:
            if args.use_ml:
                settings[test_name]['use_ml'] = True
            
            # Extract test parameters
            test_params = settings[test_name]
            
            grid = test_params['grid']
            group = test_params['group']
            top = test_params['top']
            multiple = test_params.get('multiple', 1)
            ml_enabled = test_params.get('use_ml', False)
            print(ml_enabled)
            print('Test: {0}, Grid: {1}, Group: {2}, Top {3}, Multiple {4}'.format(test_name, grid, group, top, multiple))
            
            if ml_enabled:
                print(f"Using ML potentials for short-range interactions with model: {args.ml_model}")
            
            ions_count = {}
            if lib == 'SrTiO':
                ions_count = {'Sr': 1 * multiple, 'Ti': 1 * multiple, 'O': 3 * multiple}
            elif lib == 'Y2O3':
                ions_count = {'Y': 2 * multiple, 'O': 3 * multiple}
            elif lib == 'MgAl2O4':
                ions_count = {'Mg': 1 * multiple, 'Al': 2 * multiple, 'O': 4 * multiple}
            elif lib == 'Ca3Al2Si3O12':
                ions_count = {'Ca': 3 * multiple, 'Al': 2 * multiple, 'Si': 3 * multiple, 'O': 12 * multiple}
            else:
                continue
            
            # Initialize the phase with ML potential settings if required
            phase = Phase(lib, 
                         use_ml_potential=ml_enabled, 
                         ml_model_name=args.ml_model)

            allocator = Allocate(ions_count, grid, 5.0, phase)
            
            # Run the optimization
            start_time = time()
            results, runtime, _ = allocator.optimize_cube_symmetry_ase(
                group=group,
                PoolSolutions=top,
                TimeLimit=0)
            
            # Process the results
            best_energy = process_results(lib=lib, results=results,
                                        ions_count=ions_count, test_name=test_name)
            
            # Get reference energy from CIF file
            ref_filename = f"SrTiO3.cif"
            energy = get_cif_energies(filename=ref_filename)
            if multiple > 1:
                energy = energy * multiple
                print(f"For multiple {multiple} the reference energy is {energy} eV")
            
            end_time = time()
            print(f"Total runtime: {end_time - start_time} seconds")
            
            # Add to the results dataframe
            df_summary = pd.concat([df_summary, pd.DataFrame([{
                'name': test_name,
                'grid': grid,
                'group': group,
                'best_E': best_energy,
                'expected_E': energy,
                'time': runtime,
                'ml_enabled': ml_enabled
            }])], ignore_index=True)
            
    # Save results to file
    print("\nSummary of results:")
    print(tabulate(df_summary, headers="keys", tablefmt='github', showindex=False))
    
    # Save to file
    with open(os.path.join(output_dir, "results", "summary.txt"), "w+") as f:
        print("IPCSP optimization results:", file=f)
        if args.use_ml:
            print(f"Using ML potentials with model: {args.ml_model}", file=f)
        print(tabulate(df_summary, headers="keys", tablefmt='github', showindex=False), file=f)
        
    return df_summary


if __name__ == "__main__":
    benchmark()
