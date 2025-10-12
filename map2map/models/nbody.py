import torch
from .styled_vnet import StyledVNet

class NbodyD2DStyledVNet(StyledVNet):
    def __init__(self, style_size, in_chan, out_chan, bypass=None, **kwargs):
        """Nbody ZA (linear theory) displacements to Nbody nonlinear displacements
           V-Net like network with styles
           See `vnet.VNet`.
        """
        super(NbodyD2DStyledVNet, self).__init__(style_size, in_chan, out_chan, bypass, **kwargs)


    def forward(self, x, Om, Dz):

        # Construct the style parameters
        s0 = (Om - 0.3) * 5.
        s1 = (Dz - 1.)
        s = torch.stack([s0, s1], dim=-1)

        # Rescale the ZA field
        Dz_bc = Dz.view(-1, 1, 1, 1, 1) if Dz.dim() == 1 else Dz
        x = x * Dz_bc

        x = super().forward(x, s)

        return x, s
