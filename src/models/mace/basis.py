import torch
import torch.nn as nn
import math
from e3nn import o3


class PolynomialEnvelope(nn.Module):
    """
    Polynomial envelope that smoothly goes to zero at the cutoff radius,
    ensuring continuity of the potential and its derivatives at r_cut.

    Uses the same degree-5 Behler-type polynomial as the ACE basis:
        f(u) = 1 - 6u^5 + 15u^4 - 10u^3,  u = r / r_cut
    so that f(0) = 1, f(1) = 0, f'(1) = 0, f''(1) = 0.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        u = r / self.cutoff
        # Clamp so that r > cutoff gives exactly 0 (not negative values)
        u = u.clamp(max=1.0)
        return 1.0 - 6.0 * u**5 + 15.0 * u**4 - 10.0 * u**3


class BesselBasis(nn.Module):
    """
    Radial embedding using normalised Spherical Bessel functions of the first
    kind, multiplied by a smooth cutoff envelope.

    The n-th basis function is:
        phi_n(r) = sqrt(2 / r_cut) * sin(n * pi * r / r_cut) / r

    This is the standard form used in MACE and NequIP. The sqrt(2/r_cut)
    factor ensures orthonormality on [0, r_cut].
    """

    def __init__(self, cutoff: float, num_radial: int = 8):
        super().__init__()
        self.cutoff = cutoff
        self.num_radial = num_radial
        self.envelope = PolynomialEnvelope(cutoff=cutoff)

        # Frequencies: pi, 2*pi, ..., num_radial*pi
        freqs = torch.arange(1, num_radial + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Edge lengths, shape [num_edges]
        Returns:
            Radial embeddings, shape [num_edges, num_radial]
        """
        r = r.unsqueeze(-1)  # [num_edges, 1]

        # Normalised spherical Bessel: sqrt(2/r_cut) * sin(n*pi*r/r_cut) / r
        # Add small epsilon to denominator to avoid 0/0 at r=0
        norm = math.sqrt(2.0 / self.cutoff)
        bessel = norm * torch.sin(self.freqs * r / self.cutoff) / (r + 1e-8)

        # Apply smooth cutoff envelope
        env = self.envelope(r)  # [num_edges, 1]  (broadcasts over num_radial)
        return bessel * env


class SphericalHarmonicsBasis(nn.Module):
    """
    Angular embedding using e3nn real spherical harmonics.
    """

    def __init__(self, l_max: int):
        super().__init__()
        self.l_max = l_max
        self.irreps_out = o3.Irreps.spherical_harmonics(l_max)

    def forward(self, edge_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_vec: 3D displacement vectors, shape [num_edges, 3]
                      (need not be unit vectors — e3nn normalises internally)
        Returns:
            Spherical harmonics, shape [num_edges, sum(2l+1)]
        """
        # normalize=True lets e3nn normalise the vector internally.
        # Do NOT pre-normalise manually; that would normalise twice.
        return o3.spherical_harmonics(
            self.irreps_out,
            edge_vec,
            normalize=True,
            normalization="component",
        )
