"""Entry point for training and evaluating the Engine PINN model."""

from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score

from engine_pinn.data.dataset import build_dataloaders, load_or_generate_dataframe
from engine_pinn.models.pinn import EnginePINN
from engine_pinn.training.loss import PINNLoss
from engine_pinn.training.trainer import Trainer
from engine_pinn.utils.config import PhysicsConfig, TrainConfig
from engine_pinn.utils.plotting import (
    plot_latent_trends,
    plot_parity_plots,
    plot_residuals_vs_inputs,
    plot_training_curves,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def evaluate_and_collect_latents(
    model: EnginePINN,
    loader: torch.utils.data.DataLoader,
    y_scaler,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference and return dataframe with latents and denormalized outputs."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_raw = batch["x_raw"].to(device)
            lhv_raw = batch["lhv_raw"].to(device)
            out = model(x, x_raw, lhv_raw)

            y_true = y_scaler.inverse_transform(batch["y"].numpy())
            y_pred = np.hstack([out["bsfc"].cpu().numpy(), out["nox"].cpu().numpy()])

            latent = out["latents"].cpu().numpy()
            x_raw_np = batch["x_raw"].cpu().numpy()
            lhv_np = batch["lhv_raw"].cpu().numpy()

            for i in range(len(x_raw_np)):
                rows.append(
                    {
                        "BMEP": x_raw_np[i, 0],
                        "H2_percentage": x_raw_np[i, 1],
                        "Spark_Ignition_Timing": x_raw_np[i, 2],
                        "Lambda": x_raw_np[i, 3],
                        "LHV": lhv_np[i, 0],
                        "BSFC_true": y_true[i, 0],
                        "NOx_true": y_true[i, 1],
                        "BSFC_pred": y_pred[i, 0],
                        "NOx_pred": y_pred[i, 1],
                        "L1_FMEP": latent[i, 0],
                        "L2_Indicated_Efficiency": latent[i, 1],
                        "L3_Temp_Potential": latent[i, 2],
                    }
                )
    return pd.DataFrame(rows)


def log_metrics(results_df: pd.DataFrame, logger: logging.Logger) -> None:
    for target in ["BSFC", "NOx"]:
        yt = results_df[f"{target}_true"].to_numpy()
        yp = results_df[f"{target}_pred"].to_numpy()
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))
        target_mape = mape(yt, yp)
        logger.info("%s -> RMSE=%.6f | R2=%.4f | MAPE=%.2f%%", target, rmse, r2, target_mape)


def main() -> None:
    setup_logging()
    logger = logging.getLogger("main")

    train_cfg = TrainConfig()
    phys_cfg = PhysicsConfig()

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_or_generate_dataframe(train_cfg.data_csv)
    dataset = build_dataloaders(
        df,
        batch_size=train_cfg.batch_size,
        test_size=train_cfg.test_size,
        val_size=train_cfg.val_size,
        random_state=train_cfg.seed,
    )

    nox_a_init = max(float(df["NOx"].max()), 1e-3)
    model = EnginePINN(nox_a_init=nox_a_init, nox_b_init=phys_cfg.nox_b_init).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = PINNLoss(lambda_phys=train_cfg.lambda_phys, lambda_mono=train_cfg.lambda_mono)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=train_cfg.checkpoint_dir,
        patience=train_cfg.patience,
    )

    history = trainer.fit(
        dataset.train_loader,
        dataset.val_loader,
        epochs_phase1=train_cfg.epochs_phase1,
        epochs_phase2=train_cfg.epochs_phase2,
    )

    train_cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        {"train_total": history.train_total, "val_total": history.val_total},
        train_cfg.plots_dir / "training_curves.png",
    )

    results_df = evaluate_and_collect_latents(model, dataset.test_loader, dataset.y_scaler, device)
    log_metrics(results_df, logger)

    results_path = train_cfg.plots_dir / "test_predictions_with_latents.csv"
    results_df.to_csv(results_path, index=False)
    plot_latent_trends(results_df, train_cfg.plots_dir / "latent_sensitivity.png")
    plot_parity_plots(results_df, train_cfg.plots_dir / "parity_plots.png")
    plot_residuals_vs_inputs(results_df, train_cfg.plots_dir / "residuals_vs_inputs.png")

    logger.info("Training complete. Results saved to: %s", results_path)


if __name__ == "__main__":
    main()
