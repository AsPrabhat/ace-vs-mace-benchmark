import torch
import torch.nn as nn
from e3nn import o3
from .basis import GaussianRadialBasis


class ACEDescriptor(nn.Module):
    """
    Computes Atomic Cluster Expansion (ACE) descriptors.

    This module creates rotationally invariant features (B-basis) from the atomic
    neighbourhood by coupling the atomic density (A-basis) with itself.

    Edge index convention (PyG standard):
        edge_index[0] = i  (center / receiver atom)
        edge_index[1] = j  (neighbor / sender atom)

    The vectors argument should be  r_j - r_i  (displacement FROM center TO
    neighbor, with PBC shifts already included).
    """

    def __init__(self, num_radial: int, l_max: int, r_cut: float):
        super().__init__()
        self.num_radial = num_radial
        self.l_max = l_max
        self.r_cut = r_cut

        # 1. Radial Basis
        self.radial = GaussianRadialBasis(num_radial, r_cut)

        # 2. Spherical Harmonics Irreps (e.g., 0e + 1o + 2e for l_max=2)
        self.sh_irreps = o3.Irreps.spherical_harmonics(l_max)

        # 3. A-basis Irreps (Atomic Density)
        # For each l, we have `num_radial` copies of the spherical harmonics.
        a_irreps_str = " + ".join(
            [f"{num_radial}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max + 1)]
        )
        self.a_irreps = o3.Irreps(a_irreps_str)

        # 4. B-basis (Invariant Features)
        # We couple the A-basis with itself to form 3-body invariants (scalars -> "0e").
        # FullyConnectedTensorProduct learns Clebsch-Gordan contractions automatically.
        self.tp = o3.FullyConnectedTensorProduct(self.a_irreps, self.a_irreps, "0e")

    def forward(self, vectors: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        """
        Args:
            vectors   : (num_edges, 3)  displacement vectors r_j - r_i
                        (from center atom i toward neighbor j, with PBC shifts)
            edge_index: (2, num_edges)  [center_i, neighbor_j]  PyG convention
            num_nodes : int             total number of atoms

        Returns:
            descriptors: (num_nodes, num_radial + tp_out_dim)  invariant features
        """
        E = vectors.shape[0]
        distances = torch.norm(vectors, dim=-1)

        # 1. Radial Basis  (E, num_radial)
        R_n = self.radial(distances)

        # 2. Spherical Harmonics  (E, sum(2l+1))
        Y_lm = o3.spherical_harmonics(
            self.sh_irreps, vectors, normalize=True, normalization="component"
        )

        # 3. Construct A-basis on edges
        Y_lm_split = Y_lm.split([2 * l + 1 for l in range(self.l_max + 1)], dim=-1)

        A_parts = []
        for Y_l in Y_lm_split:
            # Y_l: (E, 2l+1),  R_n: (E, num_radial)
            # -> (E, num_radial, 2l+1) -> flatten to (E, num_radial*(2l+1))
            A_l = R_n.unsqueeze(-1) * Y_l.unsqueeze(-2)
            A_parts.append(A_l.reshape(E, -1))

        A_edge = torch.cat(A_parts, dim=-1)  # (E, dim_a_irreps)

        # 4. Pool onto CENTER atoms (edge_index[0] = i, the receiver)
        center_nodes = edge_index[0]  # FIX: was edge_index[1] which accumulated onto j
        A_node = torch.zeros(num_nodes, A_edge.shape[1], dtype=A_edge.dtype, device=A_edge.device)
        A_node.index_add_(0, center_nodes, A_edge)

        # 5. B-basis: 3-body invariants via self-tensor-product of A_node
        invariants_3body = self.tp(A_node, A_node)

        # 2-body invariants: l=0 part of A_node (R_n summed over neighbours)
        invariants_2body = A_node[:, : self.num_radial]

        descriptors = torch.cat([invariants_2body, invariants_3body], dim=-1)
        return descriptors
