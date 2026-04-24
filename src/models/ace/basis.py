import torch
import torch.nn as nn

class CutoffEnvelope(nn.Module):
    """
    A polynomial cutoff envelope that ensures interactions smoothly decay to 0 
    at the cutoff radius.
    
    Using a standard polynomial: f_c(r) = 1 - 6(r/r_c)^5 + 15(r/r_c)^4 - 10(r/r_c)^3
    This has zero value, and zero first/second derivatives at r = r_c.
    """
    def __init__(self, r_cut: float):
        super().__init__()
        self.r_cut = r_cut
        
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # Clamp distance to avoid negative values past r_cut
        r = torch.clamp(r, max=self.r_cut)
        x = r / self.r_cut
        
        envelope = 1.0 - 6.0 * (x**5) + 15.0 * (x**4) - 10.0 * (x**3)
        return envelope

class GaussianRadialBasis(nn.Module):
    """
    A simple Gaussian radial basis function, multiplied by a cutoff envelope.
    This is highly visual and easy to understand for undergraduates.
    """
    def __init__(self, num_radial: int, r_cut: float):
        super().__init__()
        self.num_radial = num_radial
        self.r_cut = r_cut
        
        # Evenly spaced Gaussian centers between 0 and r_cut
        # We use Parameter with requires_grad=False so they move to the GPU properly
        # if the model is moved, but aren't updated during training.
        self.centers = nn.Parameter(
            torch.linspace(0, r_cut, num_radial), requires_grad=False
        )
        
        # Width of each Gaussian (sigma)
        self.sigma = nn.Parameter(
            torch.tensor([r_cut / num_radial]), requires_grad=False
        )
        
        self.envelope = CutoffEnvelope(r_cut)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        r shape: (num_edges,) or (..., 1)
        returns shape: (..., num_radial)
        """
        # Ensure r has a trailing dimension for broadcasting
        if r.dim() == 1:
            r = r.unsqueeze(-1)
            
        # Compute Gaussian: exp(-((r - mu) / sigma)^2)
        basis = torch.exp(-((r - self.centers) / self.sigma) ** 2)
        
        # Apply envelope so it goes to exactly zero at r_cut
        env = self.envelope(r)
        
        return basis * env
