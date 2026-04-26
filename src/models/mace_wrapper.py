import torch
import torch.nn as nn

from .mace.model import MACE

class MACEWrapper(nn.Module):
    """
    Wraps the local MACE model.
    The MACE model already takes a PyG Data object and returns a dict with 'energy' and 'forces'.
    We just provide this wrapper for consistency and dependency management.
    """
    def __init__(self, num_elements: int = 120, r_cut: float = 5.0, num_radial: int = 8, 
                 l_max: int = 2, num_blocks: int = 2, node_dim: int = 16):
        super().__init__()
        self.model = MACE(
            num_elements=num_elements,
            r_max=r_cut,
            num_radial=num_radial,
            l_max=l_max,
            num_blocks=num_blocks,
            node_dim=node_dim
        )
        
    def forward(self, data) -> dict:
        return self.model(data)
