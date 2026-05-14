"""
Spatial Functa sanity tests for SpatialModulatedINR.

Checks A–D from the verification spec:

  A. Different spatial cells produce different shifts.
  B. Coordinates inside the same 1-NN patch share the same latent cell.
  C. Local patch coordinates use pixel-center normalisation (0.125, 0.375, 0.625, 0.875 for 4-px patches).
  D. Full forward pass produces correct output shape and gradients flow back into φ.

Bonus:
  E. Linear(z_cell) is numerically identical to Conv2d(1×1) applied to the full φ grid.

Run with:
    ~/miniforge3/envs/pss/bin/python scripts/test_spatial_functa.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from SIREN import SpatialModulatedINR, make_normalized_pixel_grid


# ── helpers ──────────────────────────────────────────────────────────────────

def _pixel_index(row, col, W=32):
    return row * W + col


def _build_model(height=32, width=32, hidden=64, depth=4,
                 s=8, c=16, freq=10.0, base='siren'):
    return SpatialModulatedINR(
        height=height, width=width,
        hidden_features=hidden, num_layers=depth,
        latent_spatial_dim=s, latent_dim=c,
        base_inr_type=base,
        spatial_interp='nearest',
        use_local_coords=True,
        freq=freq,
        device='cpu',
        out_features=3,
    )


# ── Test A – different cells → different shifts ───────────────────────────────

def test_A_different_patches_different_shifts():
    """Pixels from different 4×4 patches must receive different z_cell (and shifts)."""
    H, W, s, c = 32, 32, 8, 16
    model = _build_model(H, W, s=s, c=c)

    phi = torch.randn(s, s, c)
    phi_flat = phi.reshape(-1, c)
    z_cell = phi_flat[model.flat_cell]          # (1024, c)
    shifts = model.modul(z_cell)                # (1024, hidden*depth)

    idx_00 = _pixel_index(0, 0)     # patch (0,0)
    idx_p1 = _pixel_index(4, 4)     # patch (1,1)
    idx_p2 = _pixel_index(8, 8)     # patch (2,2)

    # cells must differ for distinct patches
    assert not torch.allclose(z_cell[idx_00], z_cell[idx_p1]), \
        "FAIL A: patch(0,0) and patch(1,1) unexpectedly share z_cell"
    assert not torch.allclose(z_cell[idx_00], z_cell[idx_p2]), \
        "FAIL A: patch(0,0) and patch(2,2) unexpectedly share z_cell"
    assert not torch.allclose(shifts[idx_00], shifts[idx_p1]), \
        "FAIL A: patch(0,0) and patch(1,1) produce identical shifts"

    print("A PASS: different patches produce different z_cell and shifts.")


# ── Test B – same 1-NN patch → same latent cell ───────────────────────────────

def test_B_same_patch_same_cell():
    """All pixels in the same 4×4 patch must map to the same flat_cell index."""
    H, W, s = 32, 32, 8
    model = _build_model(H, W, s=s)

    fc = model.flat_cell   # (1024,)

    # patch (0,0) contains rows 0-3, cols 0-3
    patch00_indices = [_pixel_index(r, c) for r in range(4) for c in range(4)]
    cells_in_patch00 = fc[patch00_indices]
    assert (cells_in_patch00 == cells_in_patch00[0]).all(), \
        f"FAIL B: pixels in patch(0,0) disagree on cell: {cells_in_patch00.unique().tolist()}"

    # patch (0,1) = rows 0-3, cols 4-7
    patch01_indices = [_pixel_index(r, c) for r in range(4) for c in range(4, 8)]
    cells_in_patch01 = fc[patch01_indices]
    assert (cells_in_patch01 == cells_in_patch01[0]).all(), \
        "FAIL B: pixels in patch(0,1) disagree on cell"

    # patch (1,0) = rows 4-7, cols 0-3
    patch10_indices = [_pixel_index(r, c) for r in range(4, 8) for c in range(4)]
    cells_in_patch10 = fc[patch10_indices]
    assert (cells_in_patch10 == cells_in_patch10[0]).all(), \
        "FAIL B: pixels in patch(1,0) disagree on cell"

    # cross-patch cells must differ
    assert cells_in_patch00[0] != cells_in_patch01[0], \
        "FAIL B: patch(0,0) and patch(0,1) share cell index"
    assert cells_in_patch00[0] != cells_in_patch10[0], \
        "FAIL B: patch(0,0) and patch(1,0) share cell index"

    # verify pixel (0,0) → cell (0,0), (0,4) → cell (1,0), (4,0) → cell (0,1)
    # Note: flat_cell = cy * s + cx.  cx = col-cell, cy = row-cell.
    assert fc[_pixel_index(0, 0)] == 0,  "FAIL B: (r=0,c=0) should be cell 0"
    assert fc[_pixel_index(0, 4)] == 1,  "FAIL B: (r=0,c=4) should be cell 1"
    assert fc[_pixel_index(4, 0)] == 8,  "FAIL B: (r=4,c=0) should be cell 8  (cy=1, cx=0)"

    print("B PASS: 1-NN cell assignment is correct for all patches.")


# ── Test C – local patch coordinates ─────────────────────────────────────────

def test_C_local_patch_coords():
    """
    For 32×32 image and 8×8 grid the patch size is 4.
    Local x-coords for the first 4 columns (within patch 0) should be:
        pixel-col 0 → local_x = 0.125
        pixel-col 1 → local_x = 0.375
        pixel-col 2 → local_x = 0.625
        pixel-col 3 → local_x = 0.875
    """
    H, W, s = 32, 32, 8
    model = _build_model(H, W, s=s)

    lc = model.local_coords   # (1024, 2)  columns = [lx, ly]
    patch_size = H // s       # 4

    expected = torch.tensor([(i + 0.5) / patch_size for i in range(patch_size)])

    for col_in_patch in range(patch_size):
        global_col = col_in_patch                           # first patch
        global_row = 0
        idx = _pixel_index(global_row, global_col)
        actual_lx = lc[idx, 0].item()
        exp_lx    = expected[col_in_patch].item()
        assert abs(actual_lx - exp_lx) < 1e-6, \
            f"FAIL C: col_in_patch={col_in_patch} → local_x={actual_lx:.6f}, expected {exp_lx:.6f}"

    for row_in_patch in range(patch_size):
        global_row = row_in_patch                           # first patch
        global_col = 0
        idx = _pixel_index(global_row, global_col)
        actual_ly = lc[idx, 1].item()
        exp_ly    = expected[row_in_patch].item()
        assert abs(actual_ly - exp_ly) < 1e-6, \
            f"FAIL C: row_in_patch={row_in_patch} → local_y={actual_ly:.6f}, expected {exp_ly:.6f}"

    print("C PASS: local patch coordinates are pixel-center normalised (0.125, 0.375, 0.625, 0.875).")


# ── Test D – full forward pass and gradient flow ──────────────────────────────

def test_D_forward_shape_and_grad():
    """Forward must return (H*W, out_features) and gradients must reach φ."""
    H, W, s, c = 32, 32, 8, 16
    model = _build_model(H, W, s=s, c=c, hidden=256, depth=6, freq=10.0)

    phi = model.init_phi(device='cpu')
    phi.requires_grad_(True)

    out = model(phi)
    assert tuple(out.shape) == (H * W, 3), \
        f"FAIL D: expected output shape ({H*W}, 3), got {tuple(out.shape)}"

    out.sum().backward()
    assert phi.grad is not None, "FAIL D: phi.grad is None — no gradient reached phi"
    assert tuple(phi.grad.shape) == (s, s, c), \
        f"FAIL D: expected phi.grad shape ({s},{s},{c}), got {tuple(phi.grad.shape)}"
    assert phi.grad.abs().sum().item() > 0, "FAIL D: phi.grad is all-zero"

    print("D PASS: forward output shape (1024, 3), gradients flow back to phi (8, 8, 16).")


# ── Test E – Linear(z_cell) ≡ Conv2d(1×1)(φ) ─────────────────────────────────

def test_E_linear_equals_conv1x1():
    """
    Applying a shared Linear to each gathered z_cell is numerically identical
    to running nn.Conv2d(latent_dim, output_dim, kernel_size=1) over the full
    s×s latent grid.  This is what the Spatial Functa paper calls the
    1×1-convolution latent-to-modulation map λ(z).
    """
    s, c_in, c_out = 8, 16, 64
    lin  = nn.Linear(c_in, c_out, bias=True)
    conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)

    # copy weights
    conv.weight.data = lin.weight.data.clone().view(c_out, c_in, 1, 1)
    conv.bias.data   = lin.bias.data.clone()

    phi = torch.randn(s, s, c_in)
    phi_flat = phi.reshape(-1, c_in)

    # Linear path: apply independently to each cell
    out_lin = lin(phi_flat).reshape(s, s, c_out)

    # Conv path: (1, c_in, s, s) → (1, c_out, s, s) → (s, s, c_out)
    out_conv = conv(phi.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)

    max_diff = (out_lin - out_conv).abs().max().item()
    assert max_diff < 1e-5, f"FAIL E: max diff = {max_diff}"
    print(f"E PASS: max diff between Linear and Conv2d(1×1) = {max_diff:.2e}  (numerically identical).")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Spatial Functa sanity tests")
    print("=" * 60)
    test_A_different_patches_different_shifts()
    test_B_same_patch_same_cell()
    test_C_local_patch_coords()
    test_D_forward_shape_and_grad()
    test_E_linear_equals_conv1x1()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
