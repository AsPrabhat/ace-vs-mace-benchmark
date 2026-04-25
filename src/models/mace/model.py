import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.data import Data

from .basis import BesselBasis, SphericalHarmonicsBasis
from .blocks import SimpleMACEBlock


class MACE(nn.Module):
    """
    Pedagogical MACE model for predicting total energy and atomic forces.

    Architecture:
        1. Atomic embedding: Z -> initial scalar node features
        2. BesselBasis + SphericalHarmonicsBasis for edge features
        3. num_blocks x SimpleMACEBlock (equivariant message passing)
        4. Readout: scalar irreps -> MLP -> site energies -> sum
        5. Forces: F = -dE/dR  via torch.autograd.grad
    """

    def __init__(
        self,
        num_elements: int = 120,
        r_max: float = 5.0,
        num_radial: int = 8,
        l_max: int = 2,
        num_blocks: int = 2,
        node_dim: int = 16,
    ):
        super().__init__()
        self.r_max = r_max

        # 1. Atomic embedding (Z -> initial node features, pure scalars)
        self.node_embedding = nn.Embedding(num_elements, node_dim)

        # Node irreps built dynamically from l_max so that the model actually
        # uses the requested angular momentum channels.
        # e.g. l_max=2: "16x0e + 16x1o + 16x2e"
        #      l_max=3: "16x0e + 16x1o + 16x2e + 16x3o"
        irreps_parts = [
            f"{node_dim}x{l}{'e' if l % 2 == 0 else 'o'}"
            for l in range(l_max + 1)
        ]
        self.node_irreps = o3.Irreps(" + ".join(irreps_parts))

        # Initial projection: fills only the 0e component; higher-l start at zero
        self.initial_projection = o3.Linear(
            o3.Irreps(f"{node_dim}x0e"), self.node_irreps
        )

        # 2. Basis functions
        self.radial_basis = BesselBasis(cutoff=r_max, num_radial=num_radial)
        self.sh_basis = SphericalHarmonicsBasis(l_max=l_max)

        # 3. MACE Interaction Blocks
        self.blocks = nn.ModuleList(
            [
                SimpleMACEBlock(
                    node_irreps=str(self.node_irreps),
                    sh_irreps=str(self.sh_basis.irreps_out),
                    radial_dim=num_radial,
                )
                for _ in range(num_blocks)
            ]
        )

        # 4. Readout: extract only invariant scalars (0e) for energy prediction
        self.readout_linear = o3.Linear(
            self.node_irreps, o3.Irreps(f"{node_dim}x0e")
        )
        self.readout_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),  # atomic site energy
        )

    def forward(self, data: Data) -> dict:
        """
        Args:
            data: PyG Data object with fields
                  pos        (num_nodes, 3)   positions [requires_grad for forces]
                  z          (num_nodes,)      atomic numbers
                  edge_index (2, num_edges)    [center_i, neighbor_j]
                  edge_shift (num_edges, 3)    constant PBC shift vectors
                  batch      (num_nodes,)      optional graph membership

        Returns:
            dict with keys "energy" (num_graphs, 1) and "forces" (num_nodes, 3)
        """
        is_training = self.training

        # Ensure positions are differentiable for F = -dE/dR
        pos = data.pos
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_(True)

        z = data.z
        edge_index = data.edge_index
        row, col = edge_index  # row=i (center), col=j (neighbor)

        # Recompute PBC-corrected edge vectors from differentiable pos.
        # edge_shift is constant (no grad), so autograd traces through pos only.
        edge_shift = data.edge_shift.to(pos.device)
        edge_vec = pos[col] - pos[row] + edge_shift  # r_j - r_i + PBC shift
        edge_len = torch.norm(edge_vec, dim=-1)

        # Edge features
        radial_feat = self.radial_basis(edge_len)   # [num_edges, num_radial]
        sh_feat = self.sh_basis(edge_vec)            # [num_edges, dim_sh]

        # Node embeddings
        node_feat = self.node_embedding(z)           # [num_nodes, node_dim]
        node_feat = self.initial_projection(node_feat)

        # Message Passing Blocks
        for block in self.blocks:
            node_feat = block(node_feat, edge_index, radial_feat, sh_feat)

        # Readout
        inv_feat = self.readout_linear(node_feat)    # [num_nodes, node_dim]
        site_energies = self.readout_mlp(inv_feat)  # [num_nodes, 1]

        # Sum site energies per graph
        if hasattr(data, "batch") and data.batch is not None:
            # index_add_ is often faster than scatter() on CPU for this specific reduction
            num_graphs = data.num_graphs if hasattr(data, "num_graphs") else int(data.batch.max() + 1)
            total_energy = torch.zeros(num_graphs, 1, device=site_energies.device)
            total_energy.index_add_(0, data.batch, site_energies)
        else:
            total_energy = site_energies.sum(dim=0, keepdim=True)  # (1, 1)

        # Forces: F_i = -dE/dR_i
        forces = -torch.autograd.grad(
            outputs=total_energy,
            inputs=pos,
            grad_outputs=torch.ones_like(total_energy),
            create_graph=is_training,
            retain_graph=is_training,
        )[0]

        return {"energy": total_energy, "forces": forces}
