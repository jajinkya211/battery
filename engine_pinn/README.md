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

## Run in Google Colab (Step-by-Step)

1. Open a new Colab notebook.
2. Clone your repo in a cell (replace URL):
   ```bash
   !git clone https://github.com/<your-org>/<your-repo>.git
   %cd <your-repo>
   ```
3. Open and run `engine_pinn/Engine_PINN_End_to_End.ipynb` (or copy its cells).
4. Run the first setup cells in order:
   - path/sys.path setup cell
   - dependency install cell (`pip install -r engine_pinn/requirements.txt`)
5. (Optional) Upload your own CSV and set `custom_csv_path = Path('/content/your_file.csv')`.
6. Run remaining cells to train, evaluate, and generate artifacts.

The notebook now includes explicit Colab setup, dependency install, optional custom CSV path handling, and artifact download snippets.

### Colab Troubleshooting: `ModuleNotFoundError: No module named engine_pinn`

If you see this in Colab, the notebook is not running from a folder that contains the `engine_pinn/` package.

Run these cells first:

```python
!git clone https://github.com/<your-org>/<your-repo>.git
%cd <your-repo>
!ls engine_pinn/__init__.py
```

Then re-run the notebook bootstrap/import cells. The updated notebook also auto-searches common Colab locations and adds the correct parent folder to `sys.path`.

## Standalone Notebook Mode (Only `.ipynb` + data CSV)

If you want to run with **only two files**:
- `Engine_PINN_End_to_End.ipynb`
- your data CSV (with columns: `BMEP`, `H2_percentage`, `Spark_Ignition_Timing`, `Lambda`, `BSFC`, `NOx`)

Use the embedded self-run notebook cells:
1. Put both files in the same folder (or set `train_cfg.data_csv` to full CSV path).
2. Run all cells top-to-bottom.
3. The notebook installs missing deps, defines all classes/functions inline, trains, and exports artifacts.

No `engine_pinn` package import is required in this mode.
