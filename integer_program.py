import gurobipy as gb
import numpy as np
import os

from ipcsp import root_dir
from ipcsp.matrix_generator import Phase, get_Ewald
import ase
from ipcsp.grids_and_symmetry import cubic
import json
from ipcsp.lp_to_bqm import BQM

# Import matgl for machine learning potential
import matgl
from matgl.ext.ase import Calculator as MGLCalculator
from pymatgen.io.ase import AseAtomsAdaptor

griddir = root_dir / 'data/grids/'

class Allocate:

    def __init__(self, ions_count, grid_size, cell_size, phase):
        """
        Initialize the allocation problem
        
        Parameters:
        ions_count: Dictionary with ion types as keys and counts as values
        grid_size: Size of the grid for discretization
        cell_size: Physical size of the cell in Angstroms
        phase: Phase object containing parameters for the material
        """
        self.ions = ions_count
        self.phase = phase
        self.grid = grid_size
        self.cell = cell_size
        self.model = None
        
        # Load the M3GNet potential model 
        import matgl
        from matgl.ext.pymatgen import Structure2Graph
        
        # Get the potential and keep its Python reference
        self.potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        
        # Pre-create the converter with all element types
        self.converter = Structure2Graph(element_types=tuple(ions_count.keys()), cutoff=5.0)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def solution_to_Atoms(self, i, orbits):
        """
        Convert a solution from the integer program to an ASE Atoms object
        
        Parameters:
        i: Solution number
        orbits: Dictionary of orbits from symmetry
        
        Returns:
        ASE Atoms object
        """
        self.model.params.SolutionNumber = i

        grid_positions = cubic(self.grid)

        # Atoms
        symbols = ''
        positions = []

        for v in self.model.getVars():
            if v.Xn == 1:
                t, o = v.varName.split(sep='_')

                # The element itself
                positions.append(grid_positions[int(o)])
                symbols += t

                # And its orbit
                for pos in orbits[o]:
                    positions.append(grid_positions[pos])
                    symbols += t

        return ase.Atoms(symbols=symbols, scaled_positions=positions,
                         cell=[self.cell, self.cell, self.cell], pbc=True)

    def calculate_ml_energy(self, atom_i, atom_j, atom_type_i, atom_type_j):
        """
        Calculate the interatomic energy between two atoms using ML potential
        
        Parameters:
        atom_i, atom_j: Positions of atoms
        atom_type_i, atom_type_j: Types of atoms
        
        Returns:
        Energy value between the two atoms
        """
        # Create a temporary 2-atom structure
        temp_atoms = ase.Atoms(
            symbols=[atom_type_i, atom_type_j],
            positions=np.array([
                [0.0, 0.0, 0.0], 
                [float(atom_j[0] - atom_i[0]), 
                float(atom_j[1] - atom_i[1]), 
                float(atom_j[2] - atom_i[2])]
            ]),
            cell=np.array([
                [self.cell, 0.0, 0.0], 
                [0.0, self.cell, 0.0], 
                [0.0, 0.0, self.cell]
            ]),
            pbc=True
        )
        
        # Instead of using MGLCalculator directly, we should:
        # 1. Convert ASE Atoms to pymatgen Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        pmg_structure = AseAtomsAdaptor.get_structure(temp_atoms)
        
        # 2. Convert Structure to graph using Structure2Graph
        from matgl.ext.pymatgen import Structure2Graph
        # Get the list of elements in our system
        element_types = tuple(self.ions.keys())
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        g, state_attr = converter.get_graph(pmg_structure)
        
        # 3. Use the potential directly on the graph
        try:
            # Forward pass through the potential
            energy, forces, stresses, _ = self.potential(g, state_attr)
            # Extract just the energy (should be a single value for this simple system)
            pairwise_energy = energy.item()

        except Exception as e:
            # Fallback to a simple repulsive potential if the ML calculation fails
            print(f"Warning: ML energy calculation failed: {e}")
            vector = temp_atoms.positions[1] - temp_atoms.positions[0]
            distance = np.linalg.norm(vector)
            # Simple repulsive potential as fallback (1/r^2)
            pairwise_energy = 1.0 / (distance ** 2)
        
        return pairwise_energy / 2  # Divide by 2 to get per-atom energy

    def generate_ml_energy_matrix(self, ion_type_i, ion_type_j):
        """
        Generate a matrix of ML-based interatomic energies
        
        Parameters:
        ion_type_i, ion_type_j: Types of ions
        
        Returns:
        Matrix of pairwise energies
        """
        N = self.grid ** 3
        matrix = np.zeros((N, N))
        positions = cubic(self.grid) * self.cell
        
        # Import necessary packages here to ensure they're available
        import torch
        from pymatgen.core import Structure, Lattice
        from matgl.ext.pymatgen import Structure2Graph
        
        # Create a Structure2Graph converter specifically for this pair of elements
        # This ensures we have the correct element types for the converter
        converter = Structure2Graph(element_types=(ion_type_i, ion_type_j), cutoff=5.0)
        
        # Process each pair individually
        for i in range(N):
            for j in range(i+1, N):  # Only upper triangle
                # Calculate the relative position vector
                rel_pos = [
                    positions[j][0] - positions[i][0],
                    positions[j][1] - positions[i][1],
                    positions[j][2] - positions[i][2]
                ]
                
                try:
                    # Create a pymatgen Structure directly (bypass ASE)
                    # Define the lattice with plenty of vacuum to avoid periodic interactions
                    lattice = Lattice.cubic(self.cell * 2)
                    
                    # Create a Structure with the two atoms
                    structure = Structure(
                        lattice=lattice,
                        species=[ion_type_i, ion_type_j],
                        coords=[[0.0, 0.0, 0.0], rel_pos],
                        coords_are_cartesian=True
                    )
                    
                    # Convert to a graph
                    g, state_attr = converter.get_graph(structure)
                    
                    # Make sure we have torch tensors
                    if state_attr is None:
                        # If state_attr is None, create a dummy tensor of the right shape
                        state_attr = torch.zeros(1, 1)
                    
                    # Forward pass through the potential
                    with torch.no_grad():
                        # The potential may return different formats of outputs
                        # Usually (energy, forces, stress, hessian) or just energy
                        output = self.potential(g, state_attr)
                        
                        # Handle the output based on its type
                        if isinstance(output, tuple) and len(output) >= 1:
                            energy = output[0]
                        else:
                            energy = output
                        
                        # Convert to a Python float
                        if hasattr(energy, 'item'):
                            energy_value = energy.item() / 2  # Per-atom energy
                        else:
                            energy_value = float(energy) / 2
                    
                    # Assign to matrix (symmetric)
                    matrix[i, j] = energy_value
                    matrix[j, i] = energy_value
                        
                    except Exception as e:
                        # Fallback to a simple repulsive potential
                        print(f"Warning: ML energy calculation failed: {e}")
                        distance = np.linalg.norm(rel_pos)
                        # Simple repulsive potential as fallback (1/r^2)
                        energy_value = 1.0 / (distance ** 2)
                        matrix[i, j] = energy_value
                        matrix[j, i] = energy_value
            
                    return matrix

    def optimize_cube_symmetry_ase(self, group='1', PoolSolutions=1, TimeLimit=0, verbose=True):
        '''
        The function to generate an integer program and solve allocation problem using Gurobi.
        We rely on atomic simulation environment to handle allocations afterwards.
        
        Now uses ML potential for short-range interactions and Ewald for long-range.
        '''

        N = self.grid ** 3  # number
        T = len(self.ions)  # different types

        # PATH hack
        with open(os.path.join(".", griddir / 'CO{grid}G{group}.json'.format(grid=self.grid, group=group)), "r") as f:
            orbits = json.load(f)

        orb_key = list(orbits.keys())

        if verbose:
            print("Generating integer program\n")

        orb_size = [len(orbits[k]) + 1 for k in orb_key]

        # o_pos[position] = number_of_the_orbit
        o_pos = []
        for i in range(N):
            for orb, pos in orbits.items():
                if int(orb) == i:
                    o_pos.append(orb_key.index(orb))
                    break

                if i in pos:
                    o_pos.append(orb_key.index(orb))
                    break

        O = len(orbits)
        types = list(self.ions.keys())  # ordered list of elements
        counts = [self.ions[t] for t in types]

        m = gb.Model('Ion allocation in {name} with symmetry {group}'.format(name=self.phase.name, group=group))
        Vars = [[] for i in range(T)]

        # Create variables
        for i in range(O):  # iterate over all orbitals
            tmp_var = []
            for j in range(T):
                Vars[j] += [m.addVar(vtype=gb.GRB.BINARY, name=str(types[j]) + '_' + orb_key[i])]
                tmp_var += [Vars[j][-1]]
            if i == 0:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) == 1, 'first_orbit_has_ion')
            else:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) <= 1, f'one_per_orbit_{i}')

        for j in range(T):
            tmp = gb.LinExpr()
            for i in range(O):
                tmp.add(Vars[j][i], orb_size[i])
            m.addConstr(tmp == counts[j], f"number_of_ions_type_{j}")

        print("Variables and constraints were generated")
        energy = gb.QuadExpr()

        # Coulomb interaction (Ewald summation) - kept as in the original
        dist = get_Ewald(self.grid, self.cell)

        for i1 in range(N):
            for j1 in range(T):  # self-interaction
                energy.add(Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i1]] * dist[i1, i1] * self.phase.charge[types[j1]] ** 2)

            for i2 in range(i1 + 1, N):
                for j1 in range(T):  # pairwise Coulumb
                    energy.add(Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i2]] * 2 * dist[i1, i2] * self.phase.charge[
                        types[j1]] ** 2)

                    for j2 in range(j1 + 1, T):
                        energy.add(Vars[j1][o_pos[i1]] * Vars[j2][o_pos[i2]] * 2 * dist[i1, i2] * self.phase.charge[
                            types[j1]] * self.phase.charge[types[j2]])
                        energy.add(Vars[j2][o_pos[i1]] * Vars[j1][o_pos[i2]] * 2 * dist[i1, i2] * self.phase.charge[
                            types[j1]] * self.phase.charge[types[j2]])
        del dist
        print("Ewald sum contribution was added to the objective function")

        # ML-based short-range interactions (replacing Buckingham)
        print("Adding ML-based interatomic potential contributions")
        
        # For each pair of ion types, calculate ML-based energy matrix
        for j1 in range(T):
            for j2 in range(j1, T):
                ion_type_i = types[j1]
                ion_type_j = types[j2]
                
                # Get or generate ML energy matrix
                ml_energy_matrix = self.generate_ml_energy_matrix(ion_type_i, ion_type_j)
                
                # Add to objective function
                if j1 == j2:  # Same ion type
                    for i1 in range(N):
                        for i2 in range(i1, N):
                            if i1 != i2:  # Skip self-interaction
                                energy.add(Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i2]] * ml_energy_matrix[i1, i2])
                else:  # Different ion types
                    for i1 in range(N):
                        for i2 in range(N):
                            if i1 != i2:  # Skip self-interaction
                                energy.add(Vars[j1][o_pos[i1]] * Vars[j2][o_pos[i2]] * ml_energy_matrix[i1, i2])

        print("ML potential contribution was added to the objective function")
        print("Objective function was generated")

        m.setObjective(energy, gb.GRB.MINIMIZE)
        self.model = m

        if TimeLimit > 0:
            m.params.TimeLimit = TimeLimit

        if not verbose:
            m.params.OutputFlag = 0

        if PoolSolutions > 1:
            m.params.PoolSolutions = PoolSolutions
            m.params.PoolSearchMode = 2

        m.Params.NodefileStart = 1

        print("Writing model file")
        m.write("model.lp")

        m.optimize()
        runtime = m.Runtime

        if m.status == gb.GRB.CUTOFF:
            print("Cutoff! No solution with negative energy.")
            return None

        if m.status == gb.GRB.OPTIMAL or m.status == gb.GRB.TIME_LIMIT or gb.GRB.INTERRUPTED:
            print("There are", m.SolCount, "solutions")
            res = []
            for i in range(m.SolCount):
                res.append(self.solution_to_Atoms(i, orbits))

            print("\nThe optimal assignment is as follows:")
            for v in m.getVars():
                if v.x == 1:
                    print(v.varName, end=' ')
            print()
            print('Minimal energy via optimizer: %g' % m.objVal)

            if PoolSolutions > 1:
                return res, runtime, m.objVal
            else:
                return res[0:1], runtime, m.objVal

        return None

    def optimize_qubo(self, group='1', at_dwave=False, num_reads=10,
                             infinity_placement=100, infinity_orbit=100, annealing_time=200):
        """
        The function to optimise the structure on the quantum annealer
        infinity_orbit is the penalty for putting two ions on the orbit
        infinity_placement is the penalty for placing the incorrect number of ions into the structure
        
        NOTE: QUBO optimization still needs to be updated to use ML potential
        """
        import neal
        import dimod
        import dwave.embedding
        import dwave.system
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import EmbeddingComposite

        print("Running integer programming optimisation to generate a model file with the required coefficients and "
              "obtain the ground truth for the lowest energy allocation.")

        _, _, target_energy = self.optimize_cube_symmetry_ase(group=group, verbose=False)

        print("Generating quadratic unconstrained binary problem from model.lp")

        bqm_model = BQM()
        bqm_model.parse_lp("model.lp")
        bqm_model.parse_constraints()
        bqm_model.max_bound()
        bqm_model.qubofy(infinity_placement, infinity_orbit)

        np.set_printoptions(suppress=True)
        print('Five number summary of the interaction coefficients of the Ising hamiltonian:', np.percentile(
            np.array(list(bqm_model.quadratic.values())), [0, 25, 50, 75, 100], interpolation='midpoint'))

        print("The offset is equal to", bqm_model.offset)

        bqm = dimod.BinaryQuadraticModel(bqm_model.linear, bqm_model.quadratic, bqm_model.offset, dimod.BINARY)

        print("There are ", len(bqm.variables), "variables in the program")
        print("Running the Annealer")
        print("A series of readouts with the energy, allocation to the lattice positions:")

        def stoic(datum):
            counts = {}
            for k, v in datum[0].items():
                if k[0] in counts:
                    counts[k[0]] += v
                else:
                    counts[k[0]] = v
            return 'Counts: ' + str(counts)

        def simplify(datum):
            sample = {'energy': 0, 'sample': [], 'num_occurrences': 0}
            sample['energy'] = datum[1]
            sample['num_occurrences'] = int(datum[2])

            for k, v in datum[0].items():
                if v == 1:
                    sample['sample'].append(k)

            return sample

        if at_dwave:
            embedding = dwave.embedding.chimera.find_clique_embedding(len(bqm.variables), 16)
            print("The number of qubits: ", sum(len(chain) for chain in embedding.values()))
            print("The longest chain: ", max(len(chain) for chain in embedding.values()))
            exit()
            sampler = EmbeddingComposite(DWaveSampler())
            response = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time)
            min_energy = 1000000
            sol = None
            json_result = []

            i = 1
            for datum in response.data(['sample', 'energy', 'num_occurrences']):
                print(f"Readout {i}:", simplify(datum))
                i += 1
                json_result.append(simplify(datum))

                if datum.energy < min_energy:
                    sol = datum
                    min_energy = datum.energy

            with open('last_dwave.json', 'w') as f:
                json.dump(json_result, f, indent=2)

            print("The best found allocation:\n(atom specie, position on a lattice)")
            for i in sol.sample.keys():
                if sol.sample[i] == 1:
                    print(i)
            print("The lowest found energy: ", sol.energy, "Occurrences: ", sol.num_occurrences)
            return sol.energy, target_energy
        else:
            solver = neal.SimulatedAnnealingSampler()
            response = solver.sample(bqm, num_reads=num_reads)
            min_energy = 10000
            sample = 0

            i = 1
            for datum in response.data(['sample', 'energy', 'num_occurrences']):
                print(f"Readout {i}:", simplify(datum))
                i += 1

                if datum.energy < min_energy:
                    sample = datum
                    min_energy = datum.energy

            # ok, sample contains the best found allocation and we want to print it
            print("The best found allocation: (atom specie, position on a lattice)")
            for i in sample.sample.keys():
                if sample.sample[i] == 1:
                    print(i)
            print("The lowest found energy: ", sample.energy, "Occurrences: ", sample.num_occurrences)
            return sample.energy, target_energy