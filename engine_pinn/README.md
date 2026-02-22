# Engine PINN: Physics-Informed Virtual Sensing

This project implements a modular Physics-Informed Neural Network (PINN) to estimate:

- **Outputs:** Brake Specific Fuel Consumption (**BSFC**) and **NOx emissions**
- **Latents:** FMEP, Indicated Efficiency, and Temperature Potential

The architecture embeds deterministic physics equations directly in the forward pass and combines data, physics, and monotonicity terms in the loss.

## Project Layout

```
/engine_pinn/
├── data/
├── models/
├── training/
├── utils/
├── main.py
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r engine_pinn/requirements.txt
python engine_pinn/main.py
```

If `engine_pinn/data/engine_data.csv` is not present, synthetic data is generated automatically.

## Physics Formulation

- `BSFC = (3600 / LHV) * ((BMEP + FMEP) / (BMEP * eta_i))`
- `NOx = A * exp(-B / T_p)`

Where:
- `FMEP > 0` via Softplus
- `eta_i in [0.2, 0.8]` via scaled sigmoid
- `T_p > 0` via Softplus

## Training Strategy

1. **Phase 1:** freeze NOx scalars `A, B` and stabilize latent variable learning.
2. **Phase 2:** unfreeze all parameters and optimize full composite loss.

Features include early stopping, LR scheduling, model checkpointing, and latent sensitivity plots.

## Jupyter Notebook

An end-to-end notebook is included at:

- `engine_pinn/Engine_PINN_End_to_End.ipynb`

It covers data loading/synthesis, model setup, two-phase training, evaluation, and latent sensitivity plotting in one place.
