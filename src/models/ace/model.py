import torch
import torch.nn as nn
from .descriptors import ACEDescriptor


class ACEPotential(nn.Module):
    """
    A full interatomic potential mapping atomic environments to total energy.
    Wraps ACEDescriptor in a simple MLP to regress invariant features to
    atomic site energies, then sums to get the total energy.
    """

    def __init__(self, num_elements: int, num_radial: int, l_max: int, r_cut: float, hidden_dim: int = 32):
        super().__init__()

        self.descriptor = ACEDescriptor(num_radial, l_max, r_cut)
        self.node_embedding = nn.Embedding(num_elements, hidden_dim)

        # Input dimension: 2-body (num_radial) + 3-body (tp output) + species embedding
        tp_out_dim = self.descriptor.tp.irreps_out.dim
        in_features = num_radial + tp_out_dim + hidden_dim

        # SiLU (Swish) has smooth continuous derivatives — essential for stable forces.
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # single scalar energy per atom
        )

    def forward(
        self,
        vectors: torch.Tensor,
        edge_index: torch.Tensor,
        z: torch.Tensor,
        num_nodes: int,
        batch_indices=None,
    ):
        """
        Args:
            vectors      : (num_edges, 3)   PBC-corrected displacement vectors r_j - r_i
            edge_index   : (2, num_edges)   [center_i, neighbor_j]
            z            : (num_nodes,)     atomic numbers
            num_nodes    : int
            batch_indices: (num_nodes,) or None  maps nodes to graphs

        Returns:
            total_energy : (num_graphs, 1)  total energy per graph
        """
        descriptors = self.descriptor(vectors, edge_index, num_nodes)
        z_embed = self.node_embedding(z)
        features = torch.cat([descriptors, z_embed], dim=-1)
        
        site_energies = self.mlp(features)  # (num_nodes, 1)

        if batch_indices is None:
            # Single graph — return shape (1, 1) for consistency with batched case
            return site_energies.sum().reshape(1, 1)
        else:
            num_graphs = batch_indices.max().item() + 1
            total_energy = torch.zeros(num_graphs, 1, device=site_energies.device)
            total_energy.scatter_add_(0, batch_indices.unsqueeze(-1), site_energies)
            return total_energy
