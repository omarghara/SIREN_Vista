import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinActivation(torch.nn.Module): #We use this to more easily create hooks and track activation patterns
    def forward(self, x):
        return torch.sin(x)


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=2, num_freqs=64, sigma=10.0, include_input=False):
        super().__init__()
        B = torch.randn(num_freqs, in_dim) * sigma
        self.register_buffer("B", B)
        self.include_input = include_input

    @property
    def out_dim(self):
        dim = 2 * self.B.shape[0]
        if self.include_input:
            dim += self.B.shape[1]
        return dim

    def forward(self, coords):
        proj = 2.0 * math.pi * coords @ self.B.t()
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.include_input:
            feats = torch.cat([coords, feats], dim=-1)
        return feats
        
# Basic SIREN layers.
class SineAffine(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            freq: float = 30.0,
            start: bool = False,
            use_shift: bool = False,
            shift=None,
    ):
        """
        :param in_features: the dimension of input.
        :param out_features: the dimension of output.
        :param freq: the angular frequency, w0 in sin[w0(Wx + b)].
        :param start: whether the layer is at the start of a SIREN network.
        :param use_shift: whether the layer apply a shift on the affine transformation.
        :param shift: the shift vector.
        """
        super(SineAffine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.freq = freq
        self.start = start
        self.use_shift = use_shift
        self.activation = SinActivation()
        if use_shift:
            assert shift.size(0) == out_features
            self.shift = shift

        # Affine transformation.
        self.affine = nn.Linear(in_features, out_features, bias=True)
        self._init_affine()

    def _init_affine(self):
        # Initialize the parameters.
        b = 1 / self.in_features if self.start else math.sqrt(6 / self.in_features) / self.freq
        nn.init.uniform_(self.affine.weight, -b, b)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x):
        """
        Input format
        :param x: features or grids, from the previous SIREN layer. shape: (h * w, feature_dim).
        feature_dim = 2 for input layer and hidden_dim for hidden layer.
        """
        if self.use_shift:
            out = self.affine(x) + self.shift.unsqueeze(0)
            out = self.activation(self.freq * out)
        else:
            out = self.affine(x)
            out = self.activation(self.freq * out)
        return out



class SIREN(nn.Module):
    def __init__(
            self,
            hidden_features: int,
            num_layers: int,
            freq: float = 30.0,
            use_shift: bool = False,
            voxel: bool = False,
            out_features: int = 1,
            in_features: int = None,
    ):
        """
        :param hidden_features: the number of neurons in each hidden layer.
        :param num_layers: the number of hidden layers
        :param freq: the angular frequency, w0 in sin[w0(Wx + b)].
        :param use_shift: whether the layer apply a shift on the affine transformation.
        :param voxel: use 3D (voxel) SIREN.
        """
        super(SIREN, self).__init__()
        # Set parameters.
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.freq = freq
        self.use_shift = use_shift
        self.voxel = voxel
        self.out_features = out_features
        self.in_features = in_features if in_features is not None else (3 if voxel else 2)
        
        # Construct the layers.
        self.net = self._make_layers()
        self.hidden2rgb = nn.Linear(hidden_features, out_features, bias=True)
        b = math.sqrt(6 / hidden_features) / freq
        nn.init.uniform_(self.hidden2rgb.weight, -b, b)
        nn.init.zeros_(self.hidden2rgb.bias)

    def _make_layers(self):
        assert self.num_layers > 0
        layers = []
        for i in range(self.num_layers):
            in_features = self.in_features if i == 0 else self.hidden_features
            if self.use_shift:
                layers.append(
                    SineAffine(
                        in_features, self.hidden_features, self.freq, start=(i == 0),
                        use_shift=True, shift=torch.zeros(self.hidden_features, )
                    )
                )
            else:
                layers.append(
                    SineAffine(
                        in_features, self.hidden_features, self.freq, start=(i == 0), use_shift=False,
                    )
                )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input format
        :param x: coordinates of grids.
        Return format
        :return out: RGB values at the corresponding grids.
        """
        out = self.net(x)
        out = self.hidden2rgb(out)
        # Convert the output values to (0, 1).
        # out = torch.sigmoid(out)
        return out



class ModulatedSIREN(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            hidden_features: int,
            num_layers: int,
            modul_features: int,
            freq: float = 30.0,
            device='cuda',
            out_features: int = 1,
    ):
        """
        :param height: the height of input image.
        :param width: the width of input image.
        :param hidden_features: the number of neurons in each hidden layer.
        :param num_layers: the number of hidden layers.
        :param modul_features: the dimension of latent modulation.
        :param freq: the angular frequency, w0 in sin[w0(Wx + b)].
        """
        super(ModulatedSIREN, self).__init__()

        # Generate a mesh grid.
        self.height = height
        self.width = width
        self.out_features = out_features
        x, y = torch.meshgrid(torch.arange(height), torch.arange(width))
        x = x.float().view(-1).unsqueeze(0).to(device)
        y = y.float().view(-1).unsqueeze(0).to(device)
        self.meshgrid = torch.cat((x, y), dim=0).T

        # Construct the layers.
        self.siren = SIREN(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
            out_features=out_features,
        )

        # Modulation.
        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        """
        :param shift: the shift vector of all hidden layers.
        """
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers
        i = 0
        for layer in self.siren.net:
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]
            i += 1

    def forward(self, phi):
        """
        :param phi: the modulation parameter.
        :return: out: fitted result. (RGB values)
        """
        shift = self.modul(phi)
        self.assign_shift(shift=shift)
        coord = self.meshgrid.clone()
        out = self.siren(coord)
        return out


class ModulatedFourierSIREN(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            hidden_features: int,
            num_layers: int,
            modul_features: int,
            freq: float = 30.0,
            device='cuda',
            out_features: int = 1,
            fourier_num_freqs: int = 64,
            fourier_sigma: float = 10.0,
            fourier_include_input: bool = False,
    ):
        super(ModulatedFourierSIREN, self).__init__()

        self.height = height
        self.width = width
        self.out_features = out_features
        x, y = torch.meshgrid(torch.arange(height), torch.arange(width))
        x = x.float().view(-1).unsqueeze(0).to(device)
        y = y.float().view(-1).unsqueeze(0).to(device)
        self.meshgrid = torch.cat((x, y), dim=0).T

        self.fourier_num_freqs = fourier_num_freqs
        self.fourier_sigma = fourier_sigma
        self.fourier_include_input = fourier_include_input
        self.fourier = FourierFeatureEncoding(
            in_dim=2,
            num_freqs=fourier_num_freqs,
            sigma=fourier_sigma,
            include_input=fourier_include_input,
        )

        self.siren = SIREN(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
            out_features=out_features,
            in_features=self.fourier.out_dim,
        )

        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers
        i = 0
        for layer in self.siren.net:
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]
            i += 1

    def forward(self, phi):
        shift = self.modul(phi)
        self.assign_shift(shift=shift)
        coord = self.meshgrid.clone()
        coord = self.fourier(coord)
        out = self.siren(coord)
        return out
        
# Modulated SIREN for voxel grids.
class ModulatedSIREN3D(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            depth: int,
            hidden_features: int,
            num_layers: int,
            modul_features: int,
            freq: float = 30.0,
    ):
        """
        :param height: the height of input image.
        :param width: the width of input image.
        :param hidden_features: the number of neurons in each hidden layer.
        :param num_layers: the number of hidden layers.
        :param modul_features: the dimension of latent modulation.
        :param freq: the angular frequency, w0 in sin[w0(Wx + b)].
        """
        super(ModulatedSIREN3D, self).__init__()

        # Generate a mesh grid.
        self.height = height
        self.width = width
        self.depth = depth
        x, y, z = torch.meshgrid(torch.arange(height), torch.arange(width), torch.arange(depth))
        x = x.float().view(-1).unsqueeze(0).cuda()
        y = y.float().view(-1).unsqueeze(0).cuda()
        z = z.float().view(-1).unsqueeze(0).cuda()
        self.meshgrid = torch.cat((x, y, z), dim=0).T

        # Construct the layers.
        self.siren = SIREN(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
            voxel=True,
        )

        # Modulation.
        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        """
        :param shift: the shift vector of all hidden layers.
        """
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers
        i = 0
        for layer in self.siren.net:
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]
            i += 1

    def forward(self, phi):
        """
        :param phi: the modulation parameter.
        :return: out: fitted result. (RGB values)
        """
        shift = self.modul(phi)
        self.assign_shift(shift=shift)
        coord = self.meshgrid.clone()
    
        out = self.siren(coord)
        return out