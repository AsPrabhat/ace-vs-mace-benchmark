import torch
import torch.nn as nn

from .ace.model import ACEPotential


class ACEWrapper(nn.Module):
    """
    Wraps ACEPotential to provide a uniform API with MACEWrapper so that
    BenchmarkTrainer can treat both models identically.

    Accepts a PyTorch Geometric Data object with fields:
        pos        : (num_nodes, 3)  Cartesian positions (requires_grad for forces)
        edge_index : (2, num_edges)  [center_i, neighbor_j]
        edge_shift : (num_edges, 3)  constant PBC shift vectors (Å)
        batch      : (num_nodes,)    optional graph-to-node mapping
    """

    def __init__(
        self,
        num_elements: int = 120,
        num_radial: int = 8,
        l_max: int = 3,
        r_cut: float = 5.0,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.model = ACEPotential(
            num_elements=num_elements, num_radial=num_radial, l_max=l_max, r_cut=r_cut, hidden_dim=hidden_dim
        )

    def forward(self, data) -> dict:
        is_training = self.training

        # Ensure positions are differentiable for F = -dE/dR
        pos = data.pos
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_(True)

        row, col = data.edge_index  # row=i (center), col=j (neighbor)

        # Recompute PBC-corrected edge vectors from the differentiable pos so
        # that torch.autograd.grad can trace dE/dpos through edge_vec.
        # edge_shift is a constant tensor stored by the dataset.
        edge_shift = data.edge_shift.to(pos.device)
        vectors = pos[col] - pos[row] + edge_shift  # r_j - r_i + shift

        num_nodes = data.num_nodes
        batch_indices = data.batch if hasattr(data, "batch") and data.batch is not None else None

        total_energy = self.model(vectors, data.edge_index, data.z, num_nodes, batch_indices)

        # Forces: F_i = -dE/dR_i
        forces = -torch.autograd.grad(
            outputs=total_energy,
            inputs=pos,
            grad_outputs=torch.ones_like(total_energy),
            create_graph=is_training,
            retain_graph=is_training,
        )[0]

        return {"energy": total_energy, "forces": forces}
