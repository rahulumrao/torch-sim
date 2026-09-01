"""TorchSim interface for DeePMD-kit atomistic machine-learning models.

This module exposes :class:`DeepmdModel` as the TorchSim interface to DeePMD-kit models. The model implementation is provided by the ``deepmd_torchsim`` package and is re-exported here to provide a consistent import path within TorchSim.

The integration supports evaluation of DeePMD-kit potential-energy models within TorchSim simulations, including the computation of energies, atomic forces, and virial-derived stresses when requested.

If ``deepmd_torchsim`` is unavailable, importing this module emits a warning and provides a placeholder :class:`DeepmdModel` that raises the original ``ImportError`` upon instantiation.

References:
    DeePMD-kit:
        https://github.com/deepmodeling/deepmd-kit
    DeePMD-kit documentation:
        https://deepmd-kit.readthedocs.io/
"""

import traceback
import warnings
from typing import Any


try:
    from deepmd_torchsim import DeepmdModel
except ImportError as exc:
    warnings.warn(
        f"deepmd_torchsim import failed: {traceback.format_exc()}", stacklevel=2
    )

    from torch_sim.models.interface import ModelInterface

    class DeepmdModel(ModelInterface):
        """Placeholder when deepmd_torchsim is not installed."""

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Raise the original ImportError."""
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["DeepmdModel"]
