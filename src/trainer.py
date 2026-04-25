import time
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import Data


class BenchmarkTrainer:
    """
    A unified training loop for both ACE and MACE to ensure fair benchmarking.
    Tracks Energy MAE (meV/atom), Force MAE (meV/Å), and Training Time per epoch.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device="cpu",
        energy_weight=1.0,
        force_weight=100.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.energy_weight = energy_weight
        self.force_weight = force_weight

        self.metrics = []

    def _compute_loss_and_metrics(self, batch: Data):
        batch = batch.to(self.device)
        preds = self.model(batch)

        # Ground truth
        y_energy = batch.y          # (num_graphs, 1)  total energy in eV
        y_forces = batch.forces     # (num_nodes, 3)   forces in eV/Å

        # Catch missing labels early with a meaningful message.
        if y_energy is None:
            raise RuntimeError(
                "batch.y is None — energy labels are missing from the dataset. "
                "Make sure notebook 01 uses SinglePointCalculator to attach energy "
                "before writing the extxyz file (atoms.info['energy'] is dropped by ASE on write)."
            )
        if y_forces is None:
            raise RuntimeError(
                "batch.forces is None — force labels are missing from the dataset. "
                "Make sure notebook 01 uses SinglePointCalculator to attach forces "
                "before writing the extxyz file."
            )

        # Predictions
        p_energy = preds["energy"]  # (num_graphs, 1)
        p_forces = preds["forces"]  # (num_nodes, 3)

        # MSE loss for gradient stability
        loss_energy = nn.MSELoss()(p_energy, y_energy)
        loss_forces = nn.MSELoss()(p_forces, y_forces)

        loss = self.energy_weight * loss_energy + self.force_weight * loss_forces

        # --- Energy MAE per atom (correct normalisation) ---
        # Compute per-graph atom counts, then normalise each graph's energy.
        if hasattr(batch, "batch") and batch.batch is not None:
            atoms_per_graph = torch.bincount(
                batch.batch, minlength=batch.num_graphs
            ).float()  # (num_graphs,)
        else:
            atoms_per_graph = torch.tensor(
                [batch.pos.shape[0]], dtype=torch.float32, device=p_energy.device
            )

        p_e_per_atom = p_energy.squeeze(-1) / atoms_per_graph.to(p_energy.device)
        y_e_per_atom = y_energy.squeeze(-1) / atoms_per_graph.to(y_energy.device)
        mae_energy_mev_per_atom = (
            torch.abs(p_e_per_atom - y_e_per_atom).mean().item() * 1000.0
        )

        # --- Force MAE (meV/Å) ---
        mae_forces_mev = nn.L1Loss()(p_forces, y_forces).item() * 1000.0

        return loss, mae_energy_mev_per_atom, mae_forces_mev

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_e_mae = 0.0
        total_f_mae = 0.0

        start_time = time.time()
        for batch in self.train_loader:
            self.optimizer.zero_grad()

            loss, e_mae, f_mae = self._compute_loss_and_metrics(batch)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_e_mae += e_mae
            total_f_mae += f_mae

        end_time = time.time()

        n_batches = len(self.train_loader)
        return {
            "loss": total_loss / n_batches,
            "e_mae": total_e_mae / n_batches,
            "f_mae": total_f_mae / n_batches,
            "time": end_time - start_time,
        }

    # NOTE: We intentionally avoid @torch.no_grad() for evaluation because
    # both models compute forces via torch.autograd.grad() in forward().
    # In eval mode, wrappers set create_graph=False/retain_graph=False.
    def evaluate_loader(self, loader, prefix: str = "val"):
        self.model.eval()
        total_loss = 0.0
        total_e_mae = 0.0
        total_f_mae = 0.0

        for batch in loader:
            loss, e_mae, f_mae = self._compute_loss_and_metrics(batch)
            total_loss += loss.item()
            total_e_mae += e_mae
            total_f_mae += f_mae

        n_batches = len(loader)
        return {
            f"{prefix}_loss": total_loss / n_batches,
            f"{prefix}_e_mae": total_e_mae / n_batches,
            f"{prefix}_f_mae": total_f_mae / n_batches,
        }

    def validate_epoch(self):
        return self.evaluate_loader(self.val_loader, prefix="val")

    def test_epoch(self, test_loader):
        return self.evaluate_loader(test_loader, prefix="test")

    def train(self, max_epochs: int, patience: int = 10):
        best_val_loss = float("inf")
        patience_counter = 0
        total_start_time = time.time()

        for epoch in range(max_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            # Learning rate scheduling
            self.scheduler.step(val_metrics["val_loss"])

            # Combine metrics
            epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            self.metrics.append(epoch_metrics)

            print(
                f"Epoch {epoch:03d} | Time: {train_metrics['time']:.2f}s | "
                f"Train E MAE: {train_metrics['e_mae']:.2f} meV/atom | "
                f"Train F MAE: {train_metrics['f_mae']:.2f} meV/Å | "
                f"Val E MAE: {val_metrics['val_e_mae']:.2f} meV/atom | "
                f"Val F MAE: {val_metrics['val_f_mae']:.2f} meV/Å"
            )

            # Early stopping
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        total_time = time.time() - total_start_time
        print(f"Training completed in {total_time:.2f} seconds.")

        return pd.DataFrame(self.metrics)
