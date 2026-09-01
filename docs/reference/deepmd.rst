.. _deepmd:

DeePMD-kit
==========

A `torch-sim <https://github.com/TorchSim/torch-sim>`_ ``ModelInterface`` implementation
for `DeePMD-kit <https://github.com/deepmodeling/deepmd-kit>`_'s PyTorch backend.

Install
-------

The package (``DeepmdModel``, torch-sim ``ModelInterface`` wrapper) can be installed
with either ``pip`` or ``uv``:

.. code-block:: bash

    # from PyPI or a local checkout
    pip install deepmd-torchsim
    uv pip install deepmd-torchsim
    uv add deepmd-torchsim

    # editable, from a local checkout
    pip install -e .
    uv pip install -e .

Getting a working ``deepmd-kit`` backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install "deepmd-torchsim[deepmd]"
    uv pip install "deepmd-torchsim[deepmd]"

The ``deepmd`` extra pins a working ``deepmd-kit==3.1.3``, with ``torch==2.10.0``.

Usage
-----

.. code-block:: python

    import torch
    from deepmd_torchsim import DeepmdModel
    import torch_sim as ts
    from ase.build import molecule

    model = DeepmdModel(
        model_path="frozen_model.pth",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        compute_forces=True,
        compute_stress=True,
    )

    state = ts.io.atoms_to_state([molecule("H2O")], model.device, model.dtype)
    results = model(state)
    print(results["energy"])  # [n_systems]
    print(results["forces"])  # [n_atoms, 3]
    print(results["stress"])  # [n_systems, 3, 3]

For multitask/multi-domain foundation checkpoints (e.g. DPA-3), pass ``head=`` to
select which trained domain to evaluate with:

.. code-block:: python

    model = DeepmdModel(model_path="DPA-3.1-3M.pt", head="Omat24")

API
---

.. autoclass:: torch_sim.models.deepmd.DeepmdModel
    :members:
    :undoc-members:
    :show-inheritance:
