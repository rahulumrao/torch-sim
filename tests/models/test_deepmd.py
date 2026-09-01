"""Tests for the bundled DeePMD-kit integration (torch_sim.models.deepmd).

Fetches the small CH4 example model from deepmd_torchsim's own GitHub repo
(``tests/model/frozen_model.pth``) at test time, once per session, rather
than keeping a duplicate copy of the binary checked into this repo. Computes
a single-point energy/forces evaluation on the default device (CUDA if
available, else CPU; CPU is also checked when CUDA is the default) and
compares against a hardcoded reference. A mismatch does NOT fail the test --
the model having loaded and produced finite values is the actual pass/fail
gate -- it warns instead, so numerical drift is visible without breaking CI.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import warnings

import numpy as np
import pytest
import torch

import torch_sim as ts


try:
    from torch_sim.models.deepmd import DeepmdModel

    _IMPORT_ERROR: str | None = None
except ImportError:
    _IMPORT_ERROR = "deepmd_torchsim not installed"

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"deepmd_torchsim not installed: {_IMPORT_ERROR}"
)

FLOAT64_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_URL = (
    "https://raw.githubusercontent.com/rahulumrao/deepmd_torchsim/main/"
    "tests/model/frozen_model.pth"
)

# Computed once on a machine with both CPU and GPU (deepmd_torchsim's own
# tests/energy_forces_reference.json) -- CPU and GPU agreed to full float64
# precision there, so one reference is used for both devices here.
_REFERENCE = {
    "energy_eV": -1099.4124512539038,
    "forces_eV_per_A": [
        [0.0, 0.0, 0.0],
        [0.17491856887788382, 0.17491856887788393, 0.17491856887788393],
        [0.17491856887788382, -0.17491856887788393, -0.17491856887788393],
        [-0.17491856887788393, 0.17491856887788393, -0.17491856887788393],
        [-0.17491856887788393, -0.17491856887788382, 0.17491856887788393],
    ],
}


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Download the example CH4 model once per test session; skip if unreachable."""
    dest = tmp_path_factory.mktemp("deepmd") / "frozen_model.pth"
    try:
        urllib.request.urlretrieve(_MODEL_URL, dest)  # noqa: S310
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"could not download example model from {_MODEL_URL}: {exc}")
    return str(dest)


def build_system() -> ts.SimState:
    """A tetrahedral CH4 molecule, centered in a 10 A cubic box."""
    from ase import Atoms

    symbols = ["C", "H", "H", "H", "H"]
    positions = [
        [0.000000, 0.000000, 0.000000],
        [0.627581, 0.627581, 0.627581],
        [0.627581, -0.627581, -0.627581],
        [-0.627581, 0.627581, -0.627581],
        [-0.627581, -0.627581, 0.627581],
    ]
    box_size = 10.0
    atoms = Atoms(symbols=symbols, positions=positions, cell=[box_size] * 3, pbc=True)
    atoms.positions += box_size / 2  # center in the box
    return atoms


def compute(model_path: str, device: torch.device) -> dict:
    """Load the example CH4 model on ``device`` and return one single-point result."""
    model = DeepmdModel(
        model_path=model_path,
        device=device,
        dtype=FLOAT64_DTYPE,
        compute_forces=True,
        compute_stress=False,
    )
    state = ts.io.atoms_to_state([build_system()], device, FLOAT64_DTYPE)
    output = model.forward(state)
    return {
        "energy_eV": output["energy"][0].item(),
        "forces_eV_per_A": output["forces"].detach().cpu().tolist(),
    }


def compare_to_reference(label: str, result: dict) -> None:
    """Compare against the hardcoded reference; warn (don't fail) on mismatch."""
    tolerance = 1e-5
    energy_diff = abs(result["energy_eV"] - _REFERENCE["energy_eV"])
    forces_diff = (
        torch.tensor(result["forces_eV_per_A"])
        - torch.tensor(_REFERENCE["forces_eV_per_A"])
    ).abs().max().item()

    if energy_diff >= tolerance or forces_diff >= tolerance:
        with np.printoptions(precision=4, suppress=True, floatmode="fixed"):
            warnings.warn(
                f"{label} energy/forces do not match reference (tol {tolerance:.0e}):\n"
                f"  energy: computed={result['energy_eV']:.4f} eV, "
                f"reference={_REFERENCE['energy_eV']:.4f} eV, diff={energy_diff:.4f} eV\n"
                f"  forces: computed=\n{np.array(result['forces_eV_per_A'])}\n"
                f"  forces: reference=\n{np.array(_REFERENCE['forces_eV_per_A'])}\n"
                f"  max abs force diff={forces_diff:.4f} eV/A",
                stacklevel=2,
            )


def test_deepmd_energy_forces(model_path: str) -> None:
    """Model loads and produces finite energy/forces on the default device
    (CUDA if available, else CPU) and the results are compared against a
    checked-in reference.
    """
    default_result = compute(model_path, DEFAULT_DEVICE)
    assert torch.isfinite(torch.tensor(default_result["energy_eV"]))
    assert torch.isfinite(torch.tensor(default_result["forces_eV_per_A"])).all()
    compare_to_reference(DEFAULT_DEVICE.type, default_result)

    if DEFAULT_DEVICE.type == "cuda":
        cpu_result = compute(model_path, torch.device("cpu"))
        assert torch.isfinite(torch.tensor(cpu_result["energy_eV"]))
        assert torch.isfinite(torch.tensor(cpu_result["forces_eV_per_A"])).all()
        compare_to_reference("cpu", cpu_result)
