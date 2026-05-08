import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinActivation(torch.nn.Module):
    # We use this to more easily create hooks and track activation patterns.
    def forward(self, x):
        return torch.sin(x)


def make_normalized_pixel_grid(height, width, device=None):
    """Pixel-center normalized coordinates in [0, 1].

    Returns tensor of shape (H*W, 2) with columns [x, y] where:
        x_j = (j + 0.5) / W   horizontal, fast axis
        y_i = (i + 0.5) / H   vertical, slow axis
    """
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width

    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)

    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    if device is not None:
        grid = grid.to(device)

    return grid


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


# ============================================================
# Learnable Spectral Activation
# ============================================================

class LearnableSpectralActivation(nn.Module):
    """
    Learnable spectral activation:

        A(u) = u + sum_{k=1}^{K} a_k sin(2*pi*k*u)

    The harmonic coefficients a_k are learned. This gives the INR a learned
    internal spectral operator instead of using only a fixed sine activation.
    """

    def __init__(
            self,
            num_harmonics: int = 8,
            init_scale: float = 1e-3,
            include_linear: bool = True,
    ):
        super().__init__()

        self.num_harmonics = num_harmonics
        self.include_linear = include_linear

        self.coeffs = nn.Parameter(init_scale * torch.randn(num_harmonics))

        harmonics = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics)

    def forward(self, u):
        """
        u can be:
            (N, hidden)
            (B, N, hidden)

        returns same shape as u.
        """
        # u_expanded: (..., hidden, 1)
        u_expanded = u.unsqueeze(-1)

        # harmonics_shape: (1, 1, ..., K), broadcast over u dimensions
        harmonics_shape = (1,) * u.ndim + (self.num_harmonics,)
        harmonics = self.harmonics.view(harmonics_shape)

        sinus = torch.sin(2.0 * math.pi * harmonics * u_expanded)

        coeff_shape = (1,) * (sinus.ndim - 1) + (self.num_harmonics,)
        coeffs = self.coeffs.view(coeff_shape)

        harmonic_mix = (sinus * coeffs).sum(dim=-1)

        if self.include_linear:
            return u + harmonic_mix

        return harmonic_mix


# ============================================================
# Basic SIREN layers
# ============================================================

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
        :param in_features: input dimension.
        :param out_features: output dimension.
        :param freq: angular frequency, w0 in sin[w0(Wx + b)].
        :param start: whether this is the first SIREN layer.
        :param use_shift: whether the layer applies a shift on the affine transformation.
        :param shift: shift vector.
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

        self.affine = nn.Linear(in_features, out_features, bias=True)
        self._init_affine()

    def _init_affine(self):
        b = 1 / self.in_features if self.start else math.sqrt(6 / self.in_features) / self.freq
        nn.init.uniform_(self.affine.weight, -b, b)
        nn.init.zeros_(self.affine.bias)

    def apply_activation(self, z):
        return self.activation(self.freq * z)

    def forward(self, x):
        z = self.affine(x)

        if self.use_shift:
            z = z + self.shift.unsqueeze(0)

        out = self.apply_activation(z)
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
        super(SIREN, self).__init__()

        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.freq = freq
        self.use_shift = use_shift
        self.voxel = voxel
        self.out_features = out_features
        self.in_features = in_features if in_features is not None else (3 if voxel else 2)

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

            layers.append(
                SineAffine(
                    in_features=in_features,
                    out_features=self.hidden_features,
                    freq=self.freq,
                    start=(i == 0),
                    use_shift=self.use_shift,
                    shift=torch.zeros(self.hidden_features) if self.use_shift else None,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        out = self.hidden2rgb(out)
        return out


# ============================================================
# FINER layers
# ============================================================

class FinerAffine(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            freq: float = 30.0,
            start: bool = False,
            use_shift: bool = False,
            shift=None,
            first_bias_scale: float = None,
            scale_req_grad: bool = False,
    ):
        """
        FINER layer.

        FINER activation:
            sin(freq * (|z| + 1) * z)

        where:
            z = W x + b + shift
        """
        super(FinerAffine, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.freq = freq
        self.start = start
        self.use_shift = use_shift
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad = scale_req_grad

        if use_shift:
            assert shift.size(0) == out_features
            self.shift = shift

        self.affine = nn.Linear(in_features, out_features, bias=True)
        self._init_affine()

    def _init_affine(self):
        b = 1 / self.in_features if self.start else math.sqrt(6 / self.in_features) / self.freq
        nn.init.uniform_(self.affine.weight, -b, b)

        if self.start and self.first_bias_scale is not None:
            nn.init.uniform_(self.affine.bias, -self.first_bias_scale, self.first_bias_scale)
        else:
            nn.init.zeros_(self.affine.bias)

    def activation(self, z):
        if self.scale_req_grad:
            scale = torch.abs(z) + 1.0
        else:
            with torch.no_grad():
                scale = torch.abs(z) + 1.0

        return torch.sin(self.freq * scale * z)

    def apply_activation(self, z):
        return self.activation(z)

    def forward(self, x):
        z = self.affine(x)

        if self.use_shift:
            z = z + self.shift.unsqueeze(0)

        return self.apply_activation(z)


class FINER(nn.Module):
    def __init__(
            self,
            hidden_features: int,
            num_layers: int,
            freq: float = 30.0,
            use_shift: bool = False,
            voxel: bool = False,
            out_features: int = 1,
            in_features: int = None,
            first_bias_scale: float = None,
            scale_req_grad: bool = False,
    ):
        super(FINER, self).__init__()

        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.freq = freq
        self.use_shift = use_shift
        self.voxel = voxel
        self.out_features = out_features
        self.in_features = in_features if in_features is not None else (3 if voxel else 2)
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad = scale_req_grad

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

            layers.append(
                FinerAffine(
                    in_features=in_features,
                    out_features=self.hidden_features,
                    freq=self.freq,
                    start=(i == 0),
                    use_shift=self.use_shift,
                    shift=torch.zeros(self.hidden_features) if self.use_shift else None,
                    first_bias_scale=self.first_bias_scale if i == 0 else None,
                    scale_req_grad=self.scale_req_grad,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        out = self.hidden2rgb(out)
        return out


# ============================================================
# Learnable Spectral Activation network
# ============================================================

class SpectralAffine(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_shift: bool = False,
            shift=None,
            num_harmonics: int = 8,
            init_scale: float = 1e-3,
            include_linear: bool = True,
    ):
        super(SpectralAffine, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_shift = use_shift

        if use_shift:
            assert shift.size(0) == out_features
            self.shift = shift

        self.affine = nn.Linear(in_features, out_features, bias=True)

        self.activation = LearnableSpectralActivation(
            num_harmonics=num_harmonics,
            init_scale=init_scale,
            include_linear=include_linear,
        )

        self._init_affine()

    def _init_affine(self):
        # LSA starts close to identity, so Xavier is safer than SIREN's freq-scaled init.
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def apply_activation(self, z):
        return self.activation(z)

    def forward(self, x):
        z = self.affine(x)

        if self.use_shift:
            z = z + self.shift.unsqueeze(0)

        return self.apply_activation(z)


class LearnableSpectralNet(nn.Module):
    def __init__(
            self,
            hidden_features: int,
            num_layers: int,
            use_shift: bool = False,
            voxel: bool = False,
            out_features: int = 1,
            in_features: int = None,
            lsa_num_harmonics: int = 8,
            lsa_init_scale: float = 1e-3,
            lsa_include_linear: bool = True,
    ):
        super(LearnableSpectralNet, self).__init__()

        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.use_shift = use_shift
        self.voxel = voxel
        self.out_features = out_features
        self.in_features = in_features if in_features is not None else (3 if voxel else 2)

        self.lsa_num_harmonics = lsa_num_harmonics
        self.lsa_init_scale = lsa_init_scale
        self.lsa_include_linear = lsa_include_linear

        self.net = self._make_layers()

        self.hidden2rgb = nn.Linear(hidden_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.hidden2rgb.weight)
        nn.init.zeros_(self.hidden2rgb.bias)

    def _make_layers(self):
        assert self.num_layers > 0

        layers = []

        for i in range(self.num_layers):
            in_features = self.in_features if i == 0 else self.hidden_features

            layers.append(
                SpectralAffine(
                    in_features=in_features,
                    out_features=self.hidden_features,
                    use_shift=self.use_shift,
                    shift=torch.zeros(self.hidden_features) if self.use_shift else None,
                    num_harmonics=self.lsa_num_harmonics,
                    init_scale=self.lsa_init_scale,
                    include_linear=self.lsa_include_linear,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        out = self.hidden2rgb(out)
        return out


# ============================================================
# Modulated 2D models
# ============================================================

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
        super(ModulatedSIREN, self).__init__()

        self.height = height
        self.width = width
        self.out_features = out_features
        self.meshgrid = make_normalized_pixel_grid(height, width, device=device)

        self.siren = SIREN(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
            out_features=out_features,
        )

        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers

        for i, layer in enumerate(self.siren.net):
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]

    def forward(self, phi):
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
        self.meshgrid = make_normalized_pixel_grid(height, width, device=device)

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

        for i, layer in enumerate(self.siren.net):
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]

    def forward(self, phi):
        shift = self.modul(phi)
        self.assign_shift(shift=shift)

        coord = self.meshgrid.clone()
        coord = self.fourier(coord)
        out = self.siren(coord)
        return out


class ModulatedFINER(nn.Module):
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
            first_bias_scale: float = None,
            scale_req_grad: bool = False,
    ):
        super(ModulatedFINER, self).__init__()

        self.height = height
        self.width = width
        self.out_features = out_features
        self.meshgrid = make_normalized_pixel_grid(height, width, device=device)

        self.finer_first_bias_scale = first_bias_scale
        self.finer_scale_req_grad = scale_req_grad

        # Keep attribute name `siren` for compatibility with trainer/makeset/eval.
        self.siren = FINER(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
            out_features=out_features,
            first_bias_scale=first_bias_scale,
            scale_req_grad=scale_req_grad,
        )

        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers

        for i, layer in enumerate(self.siren.net):
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]

    def forward(self, phi):
        shift = self.modul(phi)
        self.assign_shift(shift=shift)

        coord = self.meshgrid.clone()
        out = self.siren(coord)
        return out


class ModulatedLSA(nn.Module):
    """
    Modulated Learnable Spectral Activation INR without Fourier input.

    Useful for ablation:
        normalized x,y -> LSA network -> RGB
    """

    def __init__(
            self,
            height: int,
            width: int,
            hidden_features: int,
            num_layers: int,
            modul_features: int,
            device='cuda',
            out_features: int = 1,
            lsa_num_harmonics: int = 8,
            lsa_init_scale: float = 1e-3,
            lsa_include_linear: bool = True,
    ):
        super(ModulatedLSA, self).__init__()

        self.height = height
        self.width = width
        self.out_features = out_features
        self.meshgrid = make_normalized_pixel_grid(height, width, device=device)

        self.lsa_num_harmonics = lsa_num_harmonics
        self.lsa_init_scale = lsa_init_scale
        self.lsa_include_linear = lsa_include_linear

        # Keep attribute name `siren` for compatibility.
        self.siren = LearnableSpectralNet(
            hidden_features=hidden_features,
            num_layers=num_layers,
            use_shift=True,
            out_features=out_features,
            in_features=2,
            lsa_num_harmonics=lsa_num_harmonics,
            lsa_init_scale=lsa_init_scale,
            lsa_include_linear=lsa_include_linear,
        )

        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers

        for i, layer in enumerate(self.siren.net):
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]

    def forward(self, phi):
        shift = self.modul(phi)
        self.assign_shift(shift=shift)

        coord = self.meshgrid.clone()
        out = self.siren(coord)
        return out


class ModulatedFourierLSA(nn.Module):
    """
    Modulated Fourier + Learnable Spectral Activation INR.

    Pipeline:
        normalized x,y
        -> random Fourier features
        -> modulated LearnableSpectralNet
        -> RGB

    This is the main new model we want to test for CIFAR.
    """

    def __init__(
            self,
            height: int,
            width: int,
            hidden_features: int,
            num_layers: int,
            modul_features: int,
            device='cuda',
            out_features: int = 1,
            fourier_num_freqs: int = 64,
            fourier_sigma: float = 10.0,
            fourier_include_input: bool = False,
            lsa_num_harmonics: int = 8,
            lsa_init_scale: float = 1e-3,
            lsa_include_linear: bool = True,
    ):
        super(ModulatedFourierLSA, self).__init__()

        self.height = height
        self.width = width
        self.out_features = out_features
        self.meshgrid = make_normalized_pixel_grid(height, width, device=device)

        self.fourier_num_freqs = fourier_num_freqs
        self.fourier_sigma = fourier_sigma
        self.fourier_include_input = fourier_include_input

        self.lsa_num_harmonics = lsa_num_harmonics
        self.lsa_init_scale = lsa_init_scale
        self.lsa_include_linear = lsa_include_linear

        self.fourier = FourierFeatureEncoding(
            in_dim=2,
            num_freqs=fourier_num_freqs,
            sigma=fourier_sigma,
            include_input=fourier_include_input,
        )

        # Keep attribute name `siren` for compatibility.
        self.siren = LearnableSpectralNet(
            hidden_features=hidden_features,
            num_layers=num_layers,
            use_shift=True,
            out_features=out_features,
            in_features=self.fourier.out_dim,
            lsa_num_harmonics=lsa_num_harmonics,
            lsa_init_scale=lsa_init_scale,
            lsa_include_linear=lsa_include_linear,
        )

        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers

        for i, layer in enumerate(self.siren.net):
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]

    def forward(self, phi):
        shift = self.modul(phi)
        self.assign_shift(shift=shift)

        coord = self.meshgrid.clone()
        coord = self.fourier(coord)
        out = self.siren(coord)
        return out


# ============================================================
# Modulated SIREN for voxel grids
# ============================================================

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
        super(ModulatedSIREN3D, self).__init__()

        self.height = height
        self.width = width
        self.depth = depth

        x, y, z = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            torch.arange(depth),
        )

        x = x.float().view(-1).unsqueeze(0).cuda()
        y = y.float().view(-1).unsqueeze(0).cuda()
        z = z.float().view(-1).unsqueeze(0).cuda()

        self.meshgrid = torch.cat((x, y, z), dim=0).T

        self.siren = SIREN(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
            voxel=True,
        )

        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers

        for i, layer in enumerate(self.siren.net):
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]

    def forward(self, phi):
        shift = self.modul(phi)
        self.assign_shift(shift=shift)

        coord = self.meshgrid.clone()
        out = self.siren(coord)
        return out