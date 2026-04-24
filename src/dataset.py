import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ase.io import read
from ase.neighborlist import neighbor_list as ase_neighbor_list
from tqdm.auto import tqdm
import numpy as np


def ase_radius_graph(atoms, cutoff: float):
    """
    Build a radius graph for a periodic ASE Atoms object using ASE's neighbor_list.

    Unlike torch.cdist, this correctly handles periodic boundary conditions (PBC)
    by including atoms in periodic images within the cutoff radius.

    Convention (PyG standard):
        edge_index[0] = i  (center / receiver atom)
        edge_index[1] = j  (neighbor / sender atom)

    Returns:
        edge_index : (2, num_edges) LongTensor  [i, j]
        edge_shift : (num_edges, 3) FloatTensor  PBC shift vectors (Å).
                     Stored as a constant — models recompute the differentiable
                     displacement as:  edge_vec = pos[j] - pos[i] + edge_shift
    """
    i_idx, j_idx, shift_vectors = ase_neighbor_list("ijS", atoms, cutoff)

    if len(i_idx) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_shift = torch.zeros((0, 3), dtype=torch.float32)
        return edge_index, edge_shift

    cell = torch.tensor(atoms.get_cell()[:], dtype=torch.float32)
    edge_index = torch.tensor(np.vstack([i_idx, j_idx]), dtype=torch.long)
    # Convert fractional shift vectors to Cartesian (Å)
    edge_shift = torch.tensor(shift_vectors, dtype=torch.float32) @ cell
    return edge_index, edge_shift


def _atoms_to_data(atoms, cutoff: float) -> Data:
    """Convert one ASE Atoms object to a PyG Data object."""
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)

    # PBC-aware graph — constant shift vectors allow autograd-safe edge_vec
    # recomputation inside the model: edge_vec = pos[j] - pos[i] + edge_shift
    edge_index, edge_shift = ase_radius_graph(atoms, cutoff)

    data = Data(z=z, pos=pos, edge_index=edge_index, edge_shift=edge_shift)

    # Energy — read from the SinglePointCalculator (atoms.info is dropped by extxyz)
    try:
        energy = atoms.get_potential_energy()
        data.y = torch.tensor([[energy]], dtype=torch.float32)
    except Exception:
        pass  # y stays None; trainer will raise a clear error

    # Forces — similarly from the calculator
    try:
        forces = atoms.get_forces()
        data.forces = torch.tensor(forces, dtype=torch.float32)
    except Exception:
        pass

    return data


class MaterialsProjectDataset(Dataset):
    """
    Dataset for loading Materials Project structures and converting them
    to uniform PyTorch Geometric Data objects suitable for BOTH ACE and MACE.

    Performance:
        All Data objects (including PBC-aware neighbor graphs) are pre-built
        once in __init__ and cached in memory.  This avoids re-running ASE's
        neighbor_list on every __getitem__ call, which would otherwise cost
        ~3 s/epoch × 50 epochs = 2.5 min of pure overhead with no ML work.

    Inherits from torch.utils.data.Dataset (not PyG's Dataset) — PyG's
    DataLoader accepts any dataset that returns Data objects via __getitem__.

    IMPORTANT — energy/force storage in extxyz:
        Structures must be written with a SinglePointCalculator so that ASE's
        extxyz writer persists energy and forces.  atoms.info["energy"] and
        atoms.arrays["forces"] are silently dropped on write.

    Data object fields:
        z          : (num_nodes,)      atomic numbers
        pos        : (num_nodes, 3)    Cartesian positions (Å)
        edge_index : (2, num_edges)    [center_i, neighbor_j]  PyG convention
        edge_shift : (num_edges, 3)    constant PBC shift vectors (Å)
        y          : (1, 1)            total energy (eV)  [if available]
        forces     : (num_nodes, 3)    DFT forces (eV/Å)  [if available]
    """

    def __init__(self, filepath: str, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

        if filepath.endswith(".json"):
            raise NotImplementedError(
                "JSON loading is not implemented. "
                "Save your data as an extxyz file via notebook 01_Data_Preparation.ipynb."
            )

        # Read all structures from file
        atoms_list = read(filepath, index=":")

        # Pre-build and cache ALL Data objects up front.
        # ASE's neighbor_list is slow Python (~3 ms/structure); running it once
        # here instead of on every __getitem__ call saves minutes per training run.
        print(f"Pre-computing graphs for {len(atoms_list)} structures (cutoff={cutoff} Å)...")
        self.data_list = [
            _atoms_to_data(atoms, cutoff)
            for atoms in tqdm(atoms_list, desc="Building graphs", leave=False)
        ]
        print(f"  Done. Dataset ready ({len(self.data_list)} graphs).")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # O(1) list lookup — no computation at training time
        return self.data_list[idx]
